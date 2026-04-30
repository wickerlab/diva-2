import os
import argparse
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from pathlib import Path
from sklearn.utils import resample
from sklearn.decomposition import TruncatedSVD
import aim
from matplotlib.lines import Line2D

# --- Modular Pipeline Imports ---
from scripts.cmeasures import compute_cmeasures
from scripts.meta_db import append_to_db

# --- Poisoner Imports ---
from scripts.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
from scripts.svm_art.svm_art_generate_metadb import ArtSvmPoisoner
from scripts.svm_biggio.svm_biggio_generate_metadb import BiggioSvmPoisoner
from scripts.svm_feature_collision.svm_featurecollision import FeatureCollisionPoisoner
from scripts.witches_brew.witches_brew_generate_metadb import WitchesBrewPoisoner

POISONER_MAP = {
    "alfa": AlfaPoisoner,
    "feature_noise": FeatureNoisePoisoner,
    "random_flip": RandomFlipPoisoner,
    "poissvm": PoisSVMPoisoner,
    "biggio": BiggioSvmPoisoner,
    "art": ArtSvmPoisoner,
    "feature_collision": FeatureCollisionPoisoner,
    "witches_brew": WitchesBrewPoisoner,
}

# --- Dataset Enumeration ---
class DatasetEnum(str, Enum):
    ENRON = "enron"
    IMDB = "imdb"
    MNIST = "mnist"
    SYNTHETIC = "synthetic"
    BREAST_CANCER = "breast_cancer"
    SPAMBASE = "spambase"
    DIABETES = "diabetes"
    SVHN = "svhn"

def plot_multimethod_confidence(dataset_name, methods, rates, ground_truths, predictions, probabilities, logger, aim_run, plot_path):
    logger.info("Generating Multi-Method DIVA Detection Plot...")
    
    df = pd.DataFrame({
        'Method': methods,
        'Rate': rates,
        'GT': ground_truths,
        'Pred': predictions,
        'Prob': probabilities
    })
    
    # Identify the methods tested (ignore any global 'clean' placeholders if they exist)
    attack_methods = sorted([m for m in df['Method'].unique() if m != 'clean'])
    
    if not attack_methods:
        return
        
    # Calculate optimal grid layout
    cols = int(np.ceil(np.sqrt(len(attack_methods))))
    rows = int(np.ceil(len(attack_methods) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, method in enumerate(attack_methods):
        ax = axes[idx]
        df_method = df[df['Method'] == method].copy()
        df_method.sort_values(by='Rate', inplace=True)
        
        # Plot continuous probability curve
        ax.plot(df_method['Rate'], df_method['Prob'], 'b-', linewidth=2, alpha=0.7)
        
        # Scatter points indicating correct/incorrect predictions
        for _, row in df_method.iterrows():
            is_correct = (row['GT'] == bool(row['Pred']))
            color = 'green' if is_correct else 'red'
            marker = 'o' if row['GT'] else 'X' # Circle for actual poisoned, X for actual clean
            
            ax.scatter(row['Rate'], row['Prob'], color=color, s=100, marker=marker, zorder=5, edgecolors='black')
        
        ax.set_title(f"{method.upper()}", fontweight='bold')
        ax.set_xlabel("Poisoning Rate")
        ax.set_ylabel("Confidence")
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # Remove empty subplots if grid isn't perfectly filled
    for i in range(len(attack_methods), len(axes)):
        fig.delaxes(axes[i])
        
    # Build a unified legend at the bottom of the figure
    custom_lines = [
        Line2D([0], [0], color='b', lw=2, alpha=0.7),
        Line2D([0], [0], color='gray', linestyle='--'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='green', markeredgecolor='black', markersize=10)
    ]
    fig.legend(custom_lines, ['Confidence Curve', 'Threshold (50%)', 'Correct (Poisoned)', 'Incorrect', 'Correct (Clean Baseline)'], 
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    
    plt.suptitle(f"DIVA Meta-Classifier Confidence: {dataset_name.upper()}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    aim_image = aim.Image(plot_path, caption=f"DIVA Multi-Method Detection: {dataset_name}")
    aim_run.track(aim_image, name='diva_detection_confidence_plot', context={'dataset': dataset_name})
    plt.close(fig)

def step1_prepare_clean_data(args, paths, logger):
    """Loads dataset, applies SVD dynamically, downsamples, and saves to clean CSV."""
    clean_csv_path = os.path.join(paths['clean_dir'], f"{args.dataset}_svd{args.truncated}_n{args.max_sample}_clean.csv")
    
    if os.path.exists(clean_csv_path):
        logger.info(f"Clean CSV already exists at {clean_csv_path}. Skipping generation.")
        return clean_csv_path

    os.makedirs(paths['clean_dir'], exist_ok=True)
    dataset_enum = DatasetEnum(args.dataset)

    # Data Loading & Binarization based on Enum
    if dataset_enum in [DatasetEnum.ENRON, DatasetEnum.IMDB]:
        f = np.load(paths['npz_path'], allow_pickle=True, encoding='latin1')
        X_raw = f['X_train'].reshape(1)[0]
        y_raw = np.where(f['Y_train'] > 0, 1, 0)
        
    elif dataset_enum == DatasetEnum.MNIST:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
        X_raw = mnist["data"].to_numpy() 
        y_raw = mnist["target"].to_numpy().astype(np.uint8)
        mask = np.isin(y_raw, [1, 7])
        X_raw, y_raw = X_raw[mask], y_raw[mask]
        y_raw = np.where(y_raw == 1, 0, 1)
    
    elif dataset_enum == DatasetEnum.SYNTHETIC:
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
        n_features_gen = max(args.truncated + 50, 100) 
        n_informative_gen = int(n_features_gen * 0.8)
        remaining_space = n_features_gen - n_informative_gen
        n_redundant_gen = np.random.randint(0, max(1, remaining_space))
        
        X_raw, y_raw = make_classification(
            n_samples=max(args.max_sample, 1500), n_classes=2, n_features=n_features_gen,
            n_informative=n_informative_gen, n_redundant=n_redundant_gen,
            n_clusters_per_class=np.random.randint(1, 3), weights=[0.55, 0.45],
            flip_y=0.01, class_sep=1.5, random_state=42
        )
        X_raw = StandardScaler().fit_transform(X_raw)
        y_raw = np.where(y_raw > 0, 1, 0)
        
    elif dataset_enum == DatasetEnum.BREAST_CANCER:
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        data = load_breast_cancer()
        X_raw = StandardScaler().fit_transform(data.data)
        y_raw = data.target
        
    elif dataset_enum == DatasetEnum.SPAMBASE:
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import StandardScaler
        data = fetch_openml('spambase', version=1, parser='auto')
        X_raw = StandardScaler().fit_transform(data.data.to_numpy())
        y_raw = data.target.astype(int).to_numpy()
        
    elif dataset_enum == DatasetEnum.DIABETES:
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import StandardScaler
        data = fetch_openml(name='diabetes', version=1, parser='auto')
        X_raw = StandardScaler().fit_transform(data.data.to_numpy())
        y_raw = np.where(data.target == 'tested_positive', 1, 0)
    elif dataset_enum == DatasetEnum.SVHN:
        import torch
        import torchvision
        import torchvision.transforms as transforms
        import torchvision.models as models
        
        os.makedirs(paths['raw_dir'], exist_ok=True)
        # Keep the filename format consistent so the WitchesBrew string-replacement works
        raw_pt_path = os.path.join(paths['raw_dir'], f"{args.dataset}_svd{args.truncated}_n{args.max_sample}.pt")
        
        if not os.path.exists(raw_pt_path):
            logger.info("Downloading SVHN Dataset...")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
            svhn_train = torchvision.datasets.SVHN(root="data", split='train', download=True, transform=transform)
            
            # Extract images and labels
            loader = torch.utils.data.DataLoader(svhn_train, batch_size=len(svhn_train), shuffle=False)
            X_all, y_all = next(iter(loader))
            
            # Create a Binary Subset (Digit 1 vs Digit 7)
            mask = (y_all == 1) | (y_all == 7)
            X_pair, y_pair = X_all[mask], y_all[mask]
            y_pair = torch.where(y_pair == 1, torch.tensor(0), torch.tensor(1))
            
            # Subsample
            indices = torch.randperm(len(y_pair))[:args.max_sample]
            torch.save({"X": X_pair[indices], "y": y_pair[indices]}, raw_pt_path)
            
        logger.info("Extracting Latent features for clean SVHN...")
        data = torch.load(raw_pt_path)
        X_images, y_labels = data["X"], data["y"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
        latent_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()
        
        latent_vectors = []
        with torch.no_grad():
            for i in range(0, len(X_images), 128):
                batch = X_images[i:i+128].to(device)
                latent_vectors.append(latent_extractor(batch).squeeze().cpu())
                
        X_dense = torch.cat(latent_vectors).numpy()
        y_raw = y_labels.numpy()
        
        # We skip SVD for image embeddings to preserve the 512D Latent topological structure
        col_names = [f"feature_{i}" for i in range(X_dense.shape[1])]
        df = pd.DataFrame(X_dense, columns=col_names)
        df['y'] = y_raw
        df.to_csv(clean_csv_path, index=False)
        
        return clean_csv_path
    else:
        raise ValueError(f"Dataset {args.dataset} logic missing.")

    # Dynamic SVD Handling
    max_svd_components = X_raw.shape[1] - 1
    actual_truncated = min(args.truncated, max_svd_components)
    
    if actual_truncated > 0 and X_raw.shape[1] > actual_truncated + 1:
        logger.info(f"Applying TruncatedSVD (n_components={actual_truncated})...")
        svd = TruncatedSVD(n_components=actual_truncated, random_state=42)
        X_dense = svd.fit_transform(X_raw)
        if 'svd_model_path' in paths:
            os.makedirs(os.path.dirname(paths['svd_model_path']), exist_ok=True)
            joblib.dump(svd, paths['svd_model_path'])
    else:
        if sp.issparse(X_raw):
            X_dense = X_raw.toarray()
        else:
            X_dense = X_raw

    if X_dense.shape[0] > args.max_sample:
        X_dense, y_raw = resample(X_dense, y_raw, n_samples=args.max_sample, stratify=y_raw, random_state=42)

    col_names = [f"feature_{i}" for i in range(X_dense.shape[1])]
    df = pd.DataFrame(X_dense, columns=col_names)
    df['y'] = y_raw
    df.to_csv(clean_csv_path, index=False)
    
    return clean_csv_path

def orchestrate_test(args):
    """Main Orchestrator tying together Data Prep, Poisoning, C-Measures, DB appending, and Aim Eval."""
    
    # Determine methods to run (default to all if none provided)
    methods_to_run = args.methods if args.methods else list(POISONER_MAP.keys())
    
    logger = logging.getLogger(f"DIVA_TEST_{args.dataset.upper()}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s'))
        logger.addHandler(ch)

    run = aim.Run(experiment=f"DIVA_{args.dataset.upper()}")
    run["hparams"] = vars(args)

    base_folder = os.path.join("data", "test", args.dataset)
    paths = {
        'npz_path': os.path.join(base_folder, f"{args.dataset}.npz"),
        'clean_dir': os.path.join(base_folder, "clean_data"),
        'raw_dir': os.path.join(base_folder, "raw_images"),
        'svd_model_path': os.path.join(base_folder, "models", f"svd_{args.truncated}.joblib"),
        'plots_dir': os.path.join(base_folder, "plots"),
        'test_db_path': "data/test_meta_database.csv"
    }
    os.makedirs(paths['plots_dir'], exist_ok=True)

    try:
        # Step 1: Clean Data Preparation
        logger.info("--- Step 1: Preparing Clean Data ---")
        clean_csv_path = step1_prepare_clean_data(args, paths, logger)
        dataname = Path(clean_csv_path).stem
        
        # Step 2: Poisoning (Loop over all selected methods)
        logger.info(f"--- Step 2: Executing Poisoning for methods: {methods_to_run} ---")
        all_generated_meta = []
        advx_range = np.arange(0.0, args.max, args.step)
        
        for method in methods_to_run:
            if method not in POISONER_MAP:
                logger.warning(f"Method '{method}' not found in POISONER_MAP. Skipping.")
                continue
                
            poisoner = POISONER_MAP[method](base_folder=base_folder)
            generated_meta = poisoner.apply_poisoning(clean_csv_path, advx_range)
            if generated_meta:
                all_generated_meta.extend(generated_meta)
        
        if not all_generated_meta:
            logger.warning("No metadata returned by any poisoners.")
            return

        # Step 3: Compute C-Measures & Append to DB
        logger.info("--- Step 3: Extracting C-Measures & Updating DB ---")
        paths_to_compute = [m["Path"] for m in all_generated_meta]
        cmeasures_df = compute_cmeasures(paths_to_compute, db_path=paths['test_db_path'], workers=args.workers)
        append_to_db(paths['test_db_path'], all_generated_meta, cmeasures_df)

        # Step 4: Load Meta-Learner & Evaluate
        logger.info("--- Step 4: Multi-Method Meta-Classifier Evaluation ---")
        clf = joblib.load(args.metalearner)

        df_test = pd.read_csv(paths['test_db_path'])
        
        # Filter for the dataset and ALL tested methods
        df_eval = df_test[(df_test['Data'] == dataname) & (df_test['Method'].isin(methods_to_run))].copy()

        if df_eval.empty:
            logger.error("Could not find generated records in the database for evaluation.")
            return

        drop_cols = ['Data', 'Path', 'Method', 'Rate', 'Is_Poisoned', 'error']
        feature_cols = [c for c in df_eval.columns if c not in drop_cols]
        
        if hasattr(clf, 'feature_names_in_'):
            missing_cols = set(clf.feature_names_in_) - set(feature_cols)
            for col in missing_cols:
                df_eval[col] = 0.0
            X_eval = df_eval[clf.feature_names_in_].fillna(0)
        else:
            X_eval = df_eval[feature_cols].fillna(0)

        y_actual = df_eval['Is_Poisoned'].values
        rates = df_eval['Rate'].values
        methods_eval = df_eval['Method'].values
        
        y_pred = clf.predict(X_eval)
        y_prob = clf.predict_proba(X_eval)[:, 1]

        # Log results cleanly
        for m, r, actual, pred, prob in zip(methods_eval, rates, y_actual, y_pred, y_prob):
            confidence = prob * 100
            correct = bool(actual) == bool(pred)
            logger.info(f"[{m.upper()}] Rate {r:.2f} | Actual: {bool(actual):<5} | Pred: {bool(pred):<5} | Conf: {confidence:5.2f}% | Correct: {correct}")
            run.track(confidence, name='detection_confidence', context={'method': m})

        # Generate & Track the massive grid plot
        method_str = "all" if not args.methods else "_".join(args.methods)
        plot_path = os.path.join(paths['plots_dir'], f"diva_detection_{args.dataset}_{method_str}.png")
        
        plot_multimethod_confidence(
            dataset_name=args.dataset, 
            methods=methods_eval, 
            rates=rates, 
            ground_truths=y_actual, 
            predictions=y_pred, 
            probabilities=y_prob, 
            logger=logger, 
            aim_run=run, 
            plot_path=plot_path
        )

        logger.info("DIVA Evaluation Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise e
    finally:
        run.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Meta-Learner tracing on Datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=[e.value for e in DatasetEnum], help="Dataset to process")
    parser.add_argument("--methods", nargs='+', type=str, default=None, choices=list(POISONER_MAP.keys()), help="Poisoning methods to test. Defaults to ALL if omitted.")
    parser.add_argument("--max_sample", type=int, default=2000, help="Max samples to retain after downsampling")
    parser.add_argument("--truncated", type=int, default=100, help="SVD truncation components")
    parser.add_argument("--step", type=float, default=0.1, help="Adversarial rate step size (e.g., 0.1 for 10%, 20%, 30%)")
    parser.add_argument("--max", type=float, default=0.41, help="Max rate step size")
    parser.add_argument("--metalearner", type=str, default="data/universal_meta_classifier_xgb.joblib", help="Path to meta-classifier")
    parser.add_argument("--description", type=str, default="", help="Description for tracking/logging purposes")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers for computing cmeasures")

    args = parser.parse_args()
    orchestrate_test(args)
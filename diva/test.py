import os
import argparse
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from enum import Enum
from pathlib import Path
from sklearn.utils import resample
from sklearn.decomposition import TruncatedSVD
import aim
from matplotlib.lines import Line2D

# --- Modular Pipeline Imports ---
from scripts.cmeasures import compute_cmeasures
from scripts.meta_db import append_to_db
from scripts.data_generator.image_fetcher import SOURCE_CONFIG

# --- Poisoner Imports ---
from scripts.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
from scripts.svm_art.svm_art_generate_metadb import ArtSvmPoisoner
from scripts.svm_biggio.svm_biggio_generate_metadb import BiggioSvmPoisoner
from scripts.svm_feature_collision.svm_featurecollision import FeatureCollisionPoisoner
from scripts.witches_brew.witches_brew_generate_metadb import WitchesBrewPoisoner
from scripts.poison_frogs.poison_frogs_generate_metadb import PoisonFrogsPoisoner
from scripts.bullseye_polytope.bullseye_polytope_generate_metadb import BullseyePolytopePoisoner

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==========================================
# Task Modality Configuration
# ==========================================
class TaskModality(str, Enum):
    TABULAR_BINARY = "tabular_binary"
    IMAGE_BINARY = "image_binary"
    IMAGE_MULTICLASS = "image_multiclass"

MODALITY_CONFIG = {
    TaskModality.TABULAR_BINARY: {
        "test_db_path": "data/test_meta_db_tabular.csv",
        "model_path": "data/meta_classifier_tabular_xgb.joblib",
        "valid_sources": ["synthetic", "breast_cancer", "spambase", "diabetes", "enron", "imdb"],
        "valid_poisoners": ["alfa_svm", "feature_noise_svm", "random_flip_svm", "feature_collision", "biggio_svm"]
    },
    TaskModality.IMAGE_BINARY: {
        "test_db_path": "data/test_meta_db_image.csv",
        "model_path": "data/meta_classifier_image_xgb.joblib",
        "valid_sources": list(SOURCE_CONFIG.keys()), # Pulls directly from image_fetcher!
        "valid_poisoners": ["witches_brew", "poison_frogs", "bullseye_polytope"]
    }
}

POISONER_MAP = {
    "alfa_svm": AlfaPoisoner,
    "feature_noise_svm": FeatureNoisePoisoner,
    "random_flip_svm": RandomFlipPoisoner,
    "poissvm_svm": PoisSVMPoisoner,
    "biggio_svm": BiggioSvmPoisoner,
    "art_svm": ArtSvmPoisoner,
    "feature_collision": FeatureCollisionPoisoner,
    "witches_brew": WitchesBrewPoisoner,
    "poison_frogs": PoisonFrogsPoisoner,
    "bullseye_polytope": BullseyePolytopePoisoner
}

def plot_multimethod_confidence(dataset_name, methods, rates, ground_truths, predictions, probabilities, logger, aim_run, plot_path):
    logger.info("Generating Multi-Method DIVA Detection Plot...")
    
    df = pd.DataFrame({
        'Method': methods,
        'Rate': rates,
        'GT': ground_truths,
        'Pred': predictions,
        'Prob': probabilities
    })
    
    attack_methods = sorted([m for m in df['Method'].unique() if m != 'clean'])
    if not attack_methods:
        return
        
    cols = int(np.ceil(np.sqrt(len(attack_methods))))
    rows = int(np.ceil(len(attack_methods) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, method in enumerate(attack_methods):
        ax = axes[idx]
        df_method = df[df['Method'] == method].copy()
        df_method.sort_values(by='Rate', inplace=True)
        
        ax.plot(df_method['Rate'], df_method['Prob'], 'b-', linewidth=2, alpha=0.7)
        
        for _, row in df_method.iterrows():
            is_correct = (row['GT'] == bool(row['Pred']))
            color = 'green' if is_correct else 'red'
            marker = 'o' if row['GT'] else 'X' 
            ax.scatter(row['Rate'], row['Prob'], color=color, s=100, marker=marker, zorder=5, edgecolors='black')
        
        ax.set_title(f"{method.upper()}", fontweight='bold')
        ax.set_xlabel("Poisoning Rate")
        ax.set_ylabel("Confidence")
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    for i in range(len(attack_methods), len(axes)):
        fig.delaxes(axes[i])
        
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
    """Loads dataset, routes by modality, applies Latent Extractor or SVD, and saves to clean CSV."""
    modality = TaskModality(args.modality)
    os.makedirs(paths['clean_dir'], exist_ok=True)
    
    if modality == TaskModality.IMAGE_BINARY:
        c0, c1 = args.class_0, args.class_1
        dataname = f"{args.source}_{c0}_vs_{c1}"
        clean_csv_path = os.path.join(paths['clean_dir'], f"{dataname}_clean.csv")
        
        if os.path.exists(clean_csv_path):
            logger.info(f"Clean CSV already exists at {clean_csv_path}. Skipping generation.")
            return clean_csv_path, dataname
            
        os.makedirs(paths['raw_dir'], exist_ok=True)
        raw_pt_path = os.path.join(paths['raw_dir'], f"{dataname}.pt")
        
        if not os.path.exists(raw_pt_path):
            logger.info(f"Downloading/Loading {args.source.upper()} Image Data...")
            cfg = SOURCE_CONFIG[args.source]
            
            trans_list = []
            if cfg.get("resize"): trans_list.append(transforms.Resize((32, 32)))
            trans_list.append(transforms.ToTensor())
            if cfg.get("grayscale"): trans_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            trans_list.append(transforms.Normalize(cfg["mean"], cfg["std"]))
            
            transform = transforms.Compose(trans_list)
            dataset = cfg["cls"](r="data", t=True, d=True, tr=transform)
            
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            X_all, y_all = next(iter(loader))
            
            mask = (y_all == c0) | (y_all == c1)
            X_pair, y_pair = X_all[mask], y_all[mask]
            y_pair = torch.where(y_pair == c0, torch.tensor(0), torch.tensor(1))
            
            indices = torch.randperm(len(y_pair))[:args.max_sample]
            torch.save({"X": X_pair[indices], "y": y_pair[indices]}, raw_pt_path)
            
        logger.info(f"Extracting Latent features for {dataname}...")
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
        col_names = [f"feature_{i}" for i in range(X_dense.shape[1])]
        df = pd.DataFrame(X_dense, columns=col_names)
        df['y'] = y_labels.numpy()
        df.to_csv(clean_csv_path, index=False)
        
        return clean_csv_path, dataname

    elif modality == TaskModality.TABULAR_BINARY:
        dataname = args.source
        clean_csv_path = os.path.join(paths['clean_dir'], f"{dataname}_svd{args.truncated}_n{args.max_sample}_clean.csv")
        
        if os.path.exists(clean_csv_path):
            logger.info(f"Clean CSV already exists at {clean_csv_path}. Skipping generation.")
            return clean_csv_path, dataname
            
        # Data Loading & Binarization based on Tabular Source
        if args.source in ["enron", "imdb"]:
            f = np.load(paths['npz_path'], allow_pickle=True, encoding='latin1')
            X_raw = f['X_train'].reshape(1)[0]
            y_raw = np.where(f['Y_train'] > 0, 1, 0)
            
        elif args.source == "synthetic":
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
            
        elif args.source == "breast_cancer":
            from sklearn.datasets import load_breast_cancer
            from sklearn.preprocessing import StandardScaler
            data = load_breast_cancer()
            X_raw = StandardScaler().fit_transform(data.data)
            y_raw = data.target
            
        elif args.source == "spambase":
            from sklearn.datasets import fetch_openml
            from sklearn.preprocessing import StandardScaler
            data = fetch_openml('spambase', version=1, parser='auto')
            X_raw = StandardScaler().fit_transform(data.data.to_numpy())
            y_raw = data.target.astype(int).to_numpy()
            
        elif args.source == "diabetes":
            from sklearn.datasets import fetch_openml
            from sklearn.preprocessing import StandardScaler
            data = fetch_openml(name='diabetes', version=1, parser='auto')
            X_raw = StandardScaler().fit_transform(data.data.to_numpy())
            y_raw = np.where(data.target == 'tested_positive', 1, 0)
        else:
            raise ValueError(f"Tabular Source {args.source} logic missing.")

        # SVD Handling
        max_svd_components = X_raw.shape[1] - 1
        actual_truncated = min(args.truncated, max_svd_components)
        
        if actual_truncated > 0 and X_raw.shape[1] > actual_truncated + 1:
            logger.info(f"Applying TruncatedSVD (n_components={actual_truncated})...")
            svd = TruncatedSVD(n_components=actual_truncated, random_state=42)
            X_dense = svd.fit_transform(X_raw)
        else:
            X_dense = X_raw.toarray() if sp.issparse(X_raw) else X_raw

        if X_dense.shape[0] > args.max_sample:
            X_dense, y_raw = resample(X_dense, y_raw, n_samples=args.max_sample, stratify=y_raw, random_state=42)

        col_names = [f"feature_{i}" for i in range(X_dense.shape[1])]
        df = pd.DataFrame(X_dense, columns=col_names)
        df['y'] = y_raw
        df.to_csv(clean_csv_path, index=False)
        
        return clean_csv_path, dataname

def orchestrate_test(args):
    config = MODALITY_CONFIG[TaskModality(args.modality)]
    
    if args.source not in config["valid_sources"]:
        raise ValueError(f"Source '{args.source}' is invalid for modality '{args.modality}'.")

    # Determine methods to run and enforce modality boundaries
    if args.methods:
        methods_to_run = [m for m in args.methods if m in config["valid_poisoners"]]
    else:
        methods_to_run = config["valid_poisoners"]
        
    if not methods_to_run:
        raise ValueError(f"No valid methods provided. Allowed for {args.modality}: {config['valid_poisoners']}")

    logger = logging.getLogger(f"DIVA_TEST_{args.source.upper()}")

    run = aim.Run(experiment=f"DIVA_TEST_{args.modality.upper()}")
    run["hparams"] = vars(args)

    base_folder = os.path.join("data", "test", args.source)
    paths = {
        'npz_path': os.path.join(base_folder, f"{args.source}.npz"),
        'clean_dir': os.path.join(base_folder, "clean_data"),
        'raw_dir': os.path.join(base_folder, "raw_images"),
        'plots_dir': os.path.join(base_folder, "plots"),
        'test_db_path': args.db_path if args.db_path else config["test_db_path"],
        'metalearner_path': args.metalearner if args.metalearner else config["model_path"]
    }
    os.makedirs(paths['plots_dir'], exist_ok=True)

    try:
        logger.info("--- Step 1: Preparing Clean Data ---")
        clean_csv_path, dataname = step1_prepare_clean_data(args, paths, logger)
        db_dataname = Path(clean_csv_path).stem
        
        logger.info(f"--- Step 2: Executing Poisoning for methods: {methods_to_run} ---")
        all_generated_meta = []
        advx_range = np.arange(0.0, args.max, args.step)
        
        for method in methods_to_run:
            poisoner = POISONER_MAP[method](base_folder=base_folder)
            generated_meta = poisoner.apply_poisoning(clean_csv_path, advx_range)
            if generated_meta:
                all_generated_meta.extend(generated_meta)
        
        if not all_generated_meta:
            logger.warning("No metadata returned by any poisoners.")
            return

        logger.info("--- Step 3: Extracting C-Measures & Updating DB ---")
        paths_to_compute = [m["Path"] for m in all_generated_meta]
        cmeasures_df = compute_cmeasures(paths_to_compute, db_path=paths['test_db_path'], workers=args.workers)
        append_to_db(paths['test_db_path'], all_generated_meta, cmeasures_df)

        logger.info("--- Step 4: Multi-Method Meta-Classifier Evaluation ---")
        clf = joblib.load(paths['metalearner_path'])
        df_test = pd.read_csv(paths['test_db_path'])
        
        df_eval = df_test[(df_test['Data'] == db_dataname) & (df_test['Method'].isin(methods_to_run))].copy()

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

        for m, r, actual, pred, prob in zip(methods_eval, rates, y_actual, y_pred, y_prob):
            confidence = prob * 100
            correct = bool(actual) == bool(pred)
            logger.info(f"[{m.upper()}] Rate {r:.2f} | Actual: {bool(actual):<5} | Pred: {bool(pred):<5} | Conf: {confidence:5.2f}% | Correct: {correct}")
            run.track(confidence, name='detection_confidence', context={'method': m})

        method_str = "all" if not args.methods else "_".join(args.methods)
        plot_path = os.path.join(paths['plots_dir'], f"diva_detection_{db_dataname}_{method_str}.png")
        
        plot_multimethod_confidence(
            dataset_name=dataname, 
            methods=methods_eval, rates=rates, ground_truths=y_actual, predictions=y_pred, probabilities=y_prob, 
            logger=logger, aim_run=run, plot_path=plot_path
        )

        logger.info("DIVA Evaluation Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise e
    finally:
        run.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Meta-Learner tracing on Datasets")
    
    # Core Modality Args
    parser.add_argument("--modality", type=str, required=True, choices=[e.value for e in TaskModality], help="The task modality to test.")
    parser.add_argument("--source", type=str, required=True, help="Specific dataset source (e.g., svhn, synthetic).")
    
    # Image Pair Targeting (New)
    parser.add_argument("--class_0", type=int, default=0, help="Class 0 target (for image modalities)")
    parser.add_argument("--class_1", type=int, default=1, help="Class 1 target (for image modalities)")
    
    # Attack Config
    parser.add_argument("--methods", nargs='+', type=str, default=None, help="Poisoning methods to test. Defaults to ALL valid ones.")
    parser.add_argument("--max_sample", type=int, default=2000, help="Max samples to retain after downsampling")
    parser.add_argument("--truncated", type=int, default=100, help="SVD truncation components (Tabular only)")
    parser.add_argument("--step", type=float, default=0.1, help="Adversarial rate step size")
    parser.add_argument("--max", type=float, default=0.41, help="Max rate step size")
    
    # Path Overrides
    parser.add_argument("--db_path", type=str, default=None, help="Override test meta-database path")
    parser.add_argument("--metalearner", type=str, default=None, help="Override path to meta-classifier")
    
    parser.add_argument("--workers", type=int, default=None, help="Number of workers for computing cmeasures")

    args = parser.parse_args()
    orchestrate_test(args)
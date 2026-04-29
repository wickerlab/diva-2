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

POISONER_MAP = {
    "alfa": AlfaPoisoner,
    "feature_noise": FeatureNoisePoisoner,
    "random_flip": RandomFlipPoisoner,
    "poissvm": PoisSVMPoisoner,
    "biggio": BiggioSvmPoisoner,
    "art": ArtSvmPoisoner,
    "feature_collision": FeatureCollisionPoisoner,
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

def generate_diva_plot(args, rates, ground_truths, predictions, probabilities, logger, aim_run, plot_path):
    logger.info("Generating Binary DIVA Detection Plot...")
    fig = plt.figure(figsize=(10, 6))
    
    # Plot continuous probability curve
    plt.plot(rates, probabilities, 'b-', label='Detection Confidence', linewidth=2, alpha=0.7)
    
    # Scatter points indicating correct/incorrect predictions
    for r, gt, pred, prob in zip(rates, ground_truths, predictions, probabilities):
        is_correct = (gt == bool(pred))
        color = 'green' if is_correct else 'red'
        marker = 'o' if gt else 'X' # Circle for actual poisoned, X for actual clean
        
        plt.scatter(r, prob, color=color, s=100, marker=marker, zorder=5, edgecolors='black')
            
    plt.title(f"DIVA Meta-Classifier Confidence ({args.method.upper()})")
    plt.xlabel("Poisoning Rate")
    plt.ylabel("Probability of Dataset being Poisoned")
    
    # Add 50% decision threshold
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision Threshold (50%)')
    
    # Custom Legend
    custom_lines = [
        Line2D([0], [0], color='b', lw=2, alpha=0.7),
        Line2D([0], [0], color='gray', linestyle='--'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='green', markeredgecolor='black', markersize=10)
    ]
    plt.legend(custom_lines, ['Confidence Curve', 'Decision Threshold', 'Correct (Was Poisoned)', 'Incorrect', 'Correct (Was Clean Baseline)'], loc='lower right')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)

    # Save to disk
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    # Track in Aim using the saved file
    aim_image = aim.Image(plot_path, caption=f"DIVA Detection: {args.dataset} via {args.method}")
    aim_run.track(aim_image, name='diva_detection_confidence_plot', context={'dataset': args.dataset})
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
    
    # 0. Setup Logging & Aim Tracking
    logger = logging.getLogger(f"DIVA_{args.method.upper()}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s'))
        logger.addHandler(ch)

    run = aim.Run(experiment=f"DIVA_{args.dataset.upper()}")
    run["hparams"] = vars(args)

    # 1. Setup Paths
    base_folder = os.path.join("data", "test", args.dataset)
    paths = {
        'npz_path': os.path.join(base_folder, f"{args.dataset}.npz"),
        'clean_dir': os.path.join(base_folder, "clean_data"),
        'svd_model_path': os.path.join(base_folder, "models", f"svd_{args.truncated}.joblib"),
        'plots_dir': os.path.join(base_folder, "plots"),
        'test_db_path': "data/test_meta_database.csv"  # Universal Test DB
    }
    os.makedirs(paths['plots_dir'], exist_ok=True)

    try:
        # Step 1: Clean Data Preparation
        logger.info("--- Step 1: Preparing Clean Data ---")
        clean_csv_path = step1_prepare_clean_data(args, paths, logger)
        dataname = Path(clean_csv_path).stem
        
        # Step 2: Poisoning
        logger.info("--- Step 2: Executing Poisoning ---")
        if args.method not in POISONER_MAP:
            raise ValueError(f"Method {args.method} not found in POISONER_MAP.")
            
        poisoner = POISONER_MAP[args.method](base_folder=base_folder)
        advx_range = np.arange(0.0, 0.41, args.step)
        
        # The base_poisoner now strictly returns a list of metadata dicts
        generated_meta = poisoner.apply_poisoning(clean_csv_path, advx_range)
        
        if not generated_meta:
            logger.warning("No metadata returned by poisoner.")
            return

        # Step 3: Compute C-Measures & Append to DB
        logger.info("--- Step 3: Extracting C-Measures & Updating DB ---")
        paths_to_compute = [m["Path"] for m in generated_meta]
        cmeasures_df = compute_cmeasures(paths_to_compute, db_path=paths['test_db_path'])
        append_to_db(paths['test_db_path'], generated_meta, cmeasures_df)

        # Step 4: Load Meta-Learner & Evaluate
        logger.info("--- Step 4: Binary Meta-Classifier Evaluation ---")
        logger.info(f"Loading Meta-Learner from {args.metalearner}")
        clf = joblib.load(args.metalearner)

        # Read the newly updated Test Database
        df_test = pd.read_csv(paths['test_db_path'])
        
        # Filter strictly for the exact dataset and method we just generated
        df_eval = df_test[(df_test['Data'] == dataname) & (df_test['Method'] == poisoner.name)].copy()
        df_eval.sort_values(by='Rate', inplace=True)

        if df_eval.empty:
            logger.error("Could not find generated records in the database for evaluation.")
            return

        drop_cols = ['Data', 'Path', 'Method', 'Rate', 'Is_Poisoned', 'error']
        feature_cols = [c for c in df_eval.columns if c not in drop_cols]
        
        # Ensure model features align
        if hasattr(clf, 'feature_names_in_'):
            missing_cols = set(clf.feature_names_in_) - set(feature_cols)
            for col in missing_cols:
                df_eval[col] = 0.0
            X_eval = df_eval[clf.feature_names_in_].fillna(0)
        else:
            X_eval = df_eval[feature_cols].fillna(0)

        y_actual = df_eval['Is_Poisoned'].values
        rates = df_eval['Rate'].values
        
        y_pred = clf.predict(X_eval)
        y_prob = clf.predict_proba(X_eval)[:, 1]

        eval_rates = []
        eval_confs = []

        for r, actual, pred, prob in zip(rates, y_actual, y_pred, y_prob):
            confidence = prob * 100
            correct = bool(actual) == bool(pred)
            logger.info(f"Rate {r:.2f} | Actual: {bool(actual):<5} | Pred: {bool(pred):<5} | Confidence: {confidence:5.2f}% | Correct: {correct}")
            
            # Aim Tracking
            run.track(confidence, name='detection_confidence', context={'rate': r, 'method': args.method})
            eval_rates.append(r)
            eval_confs.append(confidence)

        # Generate & Track Plot
        plot_path = os.path.join(paths['plots_dir'], f"diva_detection_{args.dataset}_{args.method}.png")
        generate_diva_plot(args, rates, y_actual, y_pred, y_prob, logger, run, plot_path)

        logger.info("DIVA Evaluation Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        run.close()
        raise e
    finally:
        run.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Meta-Learner tracing on Datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=[e.value for e in DatasetEnum], help="Dataset to process")
    parser.add_argument("--method", type=str, required=True, choices=list(POISONER_MAP.keys()), help="Poisoning method")
    parser.add_argument("--max_sample", type=int, default=2000, help="Max samples to retain after downsampling")
    parser.add_argument("--truncated", type=int, default=100, help="SVD truncation components")
    parser.add_argument("--step", type=float, default=0.1, help="Adversarial rate step size (e.g., 0.1 for 10%, 20%, 30%)")
    parser.add_argument("--metalearner", type=str, default="data/universal_meta_classifier_xgb.joblib", help="Path to meta-classifier")
    parser.add_argument("--description", type=str, default="", help="Description for tracking/logging purposes")

    args = parser.parse_args()
    orchestrate_test(args)
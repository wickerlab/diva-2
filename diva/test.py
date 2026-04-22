import os
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample
from aim import Run, Image
from matplotlib.lines import Line2D

# --- Import your implemented Poisoners ---
from scripts.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
from scripts.svm_falfa.svm_falfa_generate_metadb import FalfaNNPoisoner
from scripts.svm_art.svm_art_generate_metadb import ArtSvmPoisoner
from scripts.svm_biggio.svm_biggio_generate_metadb import BiggioSvmPoisoner

# --- Dataset Enumeration ---
class DatasetEnum(str, Enum):
    ENRON = "enron"
    IMDB = "imdb"
    MNIST = "mnist"
    SYNTHETIC= "synthetic"

# --- Mapping methods to their respective classes ---
POISONER_MAP = {
    "poissvm": PoisSVMPoisoner,
    "feature_noise": FeatureNoisePoisoner,
    "random_flip": RandomFlipPoisoner,
    "alfa": AlfaPoisoner,
    "falfa": FalfaNNPoisoner,
    "art": ArtSvmPoisoner,
    "biggio": BiggioSvmPoisoner
}

def setup_logger(dataset_name, method_name):
    os.makedirs("./logs", exist_ok=True)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s')
    fh = logging.FileHandler(f"./logs/pipeline_{dataset_name}_{method_name}.log")
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    return logging.getLogger(f"DIVA_{method_name.upper()}")

def step1_prepare_clean_data(args, paths, logger):
    """
    Step 1: Load dataset, binarize labels, apply TruncatedSVD(100), 
    downsample if necessary, and save to a clean CSV.
    """
    clean_csv_path = os.path.join(paths['clean_dir'], f"{args.dataset}_svd{args.truncated}_n{args.max_sample}_clean.csv")
    
    if os.path.exists(clean_csv_path):
        logger.info(f"Clean CSV already exists at {clean_csv_path}. Skipping generation.")
        return clean_csv_path

    os.makedirs(paths['clean_dir'], exist_ok=True)
    dataset_enum = DatasetEnum(args.dataset)

    # Data Loading & Binarization based on Enum
    if dataset_enum in [DatasetEnum.ENRON, DatasetEnum.IMDB]:
        logger.info(f"Loading raw {args.dataset} data from {paths['npz_path']}")
        f = np.load(paths['npz_path'], allow_pickle=True, encoding='latin1')
        X_raw = f['X_train'].reshape(1)[0]
        y_raw = np.where(f['Y_train'] > 0, 1, 0)
        
    elif dataset_enum == DatasetEnum.MNIST:
        logger.info("Fetching MNIST data from OpenML...")
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
        
        X_raw = mnist["data"].to_numpy() 
        y_raw = mnist["target"].to_numpy().astype(np.uint8)
        
        logger.info("Converting MNIST to keep only class 1 and 7 (Even=0, Odd=1)...")
        mask = np.isin(y_raw, [1, 7])
        X_raw = X_raw[mask]
        y_raw = y_raw[mask]
        y_raw = np.where(y_raw == 1, 0, 1)
    
    elif dataset_enum == DatasetEnum.SYNTHETIC:
        logger.info("Generating synthetic dataset mapping to the Metalearner training distribution...")
        from sklearn.datasets import make_classification
        from sklearn.preprocessing import StandardScaler
        
        # Ensure we generate enough features for the subsequent TruncatedSVD to work properly
        n_features_gen = max(args.truncated + 50, 100) 
        n_informative_gen = int(n_features_gen * 0.8)
        
        remaining_space = n_features_gen - n_informative_gen
        n_redundant_gen = np.random.randint(0, max(1, remaining_space))
        
        X_raw, y_raw = make_classification(
            n_samples=max(args.max_sample, 1500), # Generate at least max_sample points
            n_classes=2,
            n_features=n_features_gen,
            n_informative=n_informative_gen,
            n_redundant=n_redundant_gen,
            n_clusters_per_class=np.random.randint(1, 3),
            weights=[0.55, 0.45],  # Mimicking an intermediate balance from the training script
            flip_y=0.01,           # Base label noise
            class_sep=1.5,         # Standard separation 
            random_state=42
        )
        
        logger.info("Standardizing synthetic dataset features...")
        X_raw = StandardScaler().fit_transform(X_raw)
        y_raw = np.where(y_raw > 0, 1, 0)

        
    else:
        raise ValueError(f"Dataset {args.dataset} is registered in Enum but logic is missing.")

    logger.info(f"Applying universal TruncatedSVD (n_components={args.truncated})...")
    svd = TruncatedSVD(n_components=args.truncated, random_state=42)
    X_dense = svd.fit_transform(X_raw)

    if 'svd_model_path' in paths:
        os.makedirs(os.path.dirname(paths['svd_model_path']), exist_ok=True)
        joblib.dump(svd, paths['svd_model_path'])

    if X_dense.shape[0] > args.max_sample:
        logger.info(f"Downsampling from {X_dense.shape[0]} to {args.max_sample}...")
        X_dense, y_raw = resample(
            X_dense, y_raw, 
            n_samples=args.max_sample, 
            stratify=y_raw, 
            random_state=42
        )

    logger.info("Saving to CSV...")
    col_names = [f"feature_{i}" for i in range(X_dense.shape[1])]
    df = pd.DataFrame(X_dense, columns=col_names)
    df['y'] = y_raw
    
    df.to_csv(clean_csv_path, index=False)
    logger.info(f"Saved cleanly reduced CSV to {clean_csv_path}")
    
    return clean_csv_path

def step4_evaluate_diva(args, paths, logger, aim_run):
    logger.info("--- Step 4: Binary Meta-Classifier Evaluation ---")
    meta_db_path = os.path.join(paths['meta_db'])
    
    if not os.path.exists(meta_db_path):
        logger.error(f"MetaDB not found at {meta_db_path}. Ensure the poisoning pipeline ran successfully.")
        return

    logger.info(f"Loading Meta-Learner from {paths['metalearner_path']}")
    meta_learner = joblib.load(paths['metalearner_path'])
    
    df_meta = pd.read_csv(meta_db_path).sort_values(by="Rate")
    rates, predictions, probabilities, ground_truths = [], [], [], []

    # Exclude metadata AND old empirical accuracy columns (to ensure compatibility with optimized poisoners)
    exclude_cols = ['Data', 'Path.Poison', 'Rate', 'Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison', 'key', 'file', 'error', 'Is_Poisoned']
    feature_cols = [c for c in df_meta.columns if c not in exclude_cols]

    for _, row in df_meta.iterrows():
        rate = row['Rate']
        is_actually_poisoned = (rate > 0.0)
        
        # Prepare feature vector
        X_meta_df = pd.DataFrame([row[feature_cols]])
        X_meta = X_meta_df.fillna(0.0).values
        
        # Binary prediction (0 or 1 / True or False)
        pred_class = meta_learner.predict(X_meta)[0]
        
        # Probability confidence (if supported by the model, otherwise fallback to class output)
        if hasattr(meta_learner, "predict_proba"):
            pred_prob = meta_learner.predict_proba(X_meta)[0][1] # Probability of being Poisoned (Class 1)
        else:
            pred_prob = float(pred_class)
            
        is_correct = (bool(pred_class) == is_actually_poisoned)
        
        rates.append(rate)
        predictions.append(pred_class)
        probabilities.append(pred_prob)
        ground_truths.append(is_actually_poisoned)
        
        logger.info(f"Rate {rate:.2f} | Actual: {is_actually_poisoned} | Pred: {bool(pred_class)} | Confidence: {pred_prob:.2%} | Correct: {is_correct}")

        # Track to Aim
        step_idx = int(rate * 100) 
        aim_run.track(float(pred_class), name='predicted_class_poisoned', step=step_idx)
        aim_run.track(pred_prob, name='probability_poisoned', step=step_idx)
        aim_run.track(float(is_correct), name='prediction_correct', step=step_idx)

    generate_diva_plot(args, rates, ground_truths, predictions, probabilities, logger, aim_run)

def generate_diva_plot(args, rates, ground_truths, predictions, probabilities, logger, aim_run):
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

    aim_run.track(Image(fig), name='diva_detection_confidence_plot', context={'dataset': args.dataset})
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIVA Poisoning and Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, required=True, choices=[d.value for d in DatasetEnum], help="Dataset name")
    parser.add_argument("--method", type=str, required=True, choices=POISONER_MAP.keys(), help="Poisoning method to apply")
    parser.add_argument("--max", type=float, default=0.41, help="Max poisoning rate")
    parser.add_argument("--step", type=float, default=0.05, help="Poisoning rate step size")
    parser.add_argument("--max_sample", type=int, default=50000, help="Downsample threshold")
    parser.add_argument("--metalearner", type=str, default="metalearner_random_flip_svm.pkl", help="Name of the metalearner model")
    parser.add_argument("--truncated", type=int, default=100, help="Feature reduction")
    parser.add_argument("--max_worker", type=int, default=None, help="Max worker used to extract in parallel complexity measure")
    parser.add_argument("--description", type=str, default="", help="Description of the run for Aim")
    args = parser.parse_args()

    base_folder = f"./data/test/{args.dataset}"
    paths = {
        'base_folder': base_folder,
        'npz_path': f"{base_folder}/{args.dataset}_processed_sparse.npz",
        'clean_dir': f"{base_folder}/clean_data",
        'svd_model_path': f"{base_folder}/{args.dataset}_svd_model.pkl",
        'metalearner_path': f"./data/{args.metalearner}"
    }

    logger = setup_logger(args.dataset, args.method)

    try:
        logger.info("--- Step 1: Preparing Clean Data (SVD) ---")
        clean_file = step1_prepare_clean_data(args, paths, logger)
        
        logger.info("--- Step 2 & 3: Executing Poisoning and C-Measure Extraction ---")
        advx_range = np.arange(0.0, args.max, args.step)
        
        PoisonerClass = POISONER_MAP[args.method]
        poisoner = PoisonerClass(base_folder=paths['base_folder'])
        
        meta_db = poisoner.run_pipeline([clean_file], advx_range, entrypoint="poison", max_worker=args.max_worker)
        paths['meta_db'] = meta_db

        run = Run(experiment=f"DIVA_{args.dataset.upper()}")
        run.set("description", args.description)
        run["hparams"] = vars(args)
        
        step4_evaluate_diva(args, paths, logger, aim_run=run)
        
        logger.info("DIVA Evaluation Pipeline completed successfully.")
        run.close()
    except Exception as e:
        logger.exception("A fatal error occurred during pipeline execution:")
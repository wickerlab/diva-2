import os
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from enum import Enum

from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml, load_breast_cancer, fetch_covtype

from pymfe.mfe import MFE
from aim import Run

# Suppress harmless pymfe standard deviation RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pymfe")
warnings.filterwarnings("ignore", message="Can't summarize feature.*")

def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s')
    # Use a unified log file for the all-dataset run
    fh = logging.FileHandler("./logs/pipeline_all_clean_baseline.log")
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    return logging.getLogger("DIVA_BASELINE_ALL")

# --- Updated Dataset Enumeration ---
class DatasetEnum(str, Enum):
    ENRON = "enron"
    IMDB = "imdb"
    MNIST = "mnist"
    BREAST_CANCER = "breast_cancer"
    SPAMBASE = "spambase"
    FASHION_MNIST = "fashion_mnist"
    ELECTRICITY = "electricity"
    COVERTYPE = "covertype"
    CIFAR10 = "cifar10"

def _load_raw_data(dataset_enum, paths, logger):
    """Auxiliary function to load and binarize diverse datasets."""
    dataset_name = dataset_enum.value
    
    if dataset_enum in [DatasetEnum.ENRON, DatasetEnum.IMDB]:
        logger.info(f"Loading raw {dataset_name} data from {paths['npz_path']}")
        f = np.load(paths['npz_path'], allow_pickle=True, encoding='latin1')
        X_raw = f['X_train'].reshape(1)[0]
        y_raw = np.where(f['Y_train'] > 0, 1, 0)

    elif dataset_enum == DatasetEnum.MNIST:
        logger.info("Fetching MNIST data from OpenML...")
        mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
        X_raw = mnist["data"].to_numpy() 
        y_raw = mnist["target"].to_numpy().astype(np.uint8)
        mask = np.isin(y_raw, [1, 7])
        X_raw, y_raw = X_raw[mask], y_raw[mask]
        y_raw = np.where(y_raw == 1, 0, 1)

    elif dataset_enum == DatasetEnum.BREAST_CANCER:
        logger.info("Loading Breast Cancer dataset (Easy, Tabular)...")
        data = load_breast_cancer()
        X_raw, y_raw = data.data, data.target

    elif dataset_enum == DatasetEnum.SPAMBASE:
        logger.info("Fetching Spambase dataset from OpenML (Medium, Tabular)...")
        data = fetch_openml('spambase', version=1, cache=True, parser='auto')
        X_raw = data.data.to_numpy()
        y_raw = data.target.to_numpy().astype(np.uint8)

    elif dataset_enum == DatasetEnum.FASHION_MNIST:
        logger.info("Fetching Fashion-MNIST from OpenML (Medium, Image)...")
        data = fetch_openml('Fashion-MNIST', version=1, cache=True, parser='auto')
        X_raw = data.data.to_numpy()
        y_raw = data.target.to_numpy().astype(np.uint8)
        # Classify T-shirt/top (0) vs Trouser (1)
        mask = np.isin(y_raw, [0, 1])
        X_raw, y_raw = X_raw[mask], y_raw[mask]

    elif dataset_enum == DatasetEnum.ELECTRICITY:
        logger.info("Fetching Electricity dataset from OpenML (Medium, Tabular)...")
        data = fetch_openml('electricity', version=1, cache=True, parser='auto')
        # Electricity has categorical features; we drop them for pure numerical SVD pipeline
        df = data.data.select_dtypes(include=[np.number])
        X_raw = df.to_numpy()
        y_raw = np.where(data.target == 'UP', 1, 0)

    elif dataset_enum == DatasetEnum.COVERTYPE:
        logger.info("Loading Forest Covertype dataset (Hard, Tabular)...")
        data = fetch_covtype()
        X_raw = data.data
        # Target is 1-7. Binarize: Lodgepole Pine (Class 2) vs Rest
        y_raw = np.where(data.target == 2, 1, 0)

    elif dataset_enum == DatasetEnum.CIFAR10:
        logger.info("Fetching CIFAR-10 from OpenML (Hard, High-Dim Image)...")
        data = fetch_openml('CIFAR_10', version=1, cache=True, parser='auto')
        X_raw = data.data.to_numpy()
        y_raw = data.target.to_numpy().astype(np.uint8)
        # Binarize: Vehicles/Objects (0-4) vs Animals (5-9)
        y_raw = np.where(y_raw >= 5, 1, 0)

    else:
        raise ValueError(f"Dataset {dataset_name} is registered but logic is missing.")

    # Handle NaNs from OpenML if they exist
    if np.isnan(X_raw).any():
        logger.info("Imputing NaNs with 0...")
        X_raw = np.nan_to_num(X_raw)

    return X_raw, y_raw

def step1_prepare_clean_data(dataset_enum, args, paths, logger):
    """
    Step 1: Load dataset, binarize labels, apply TruncatedSVD safely, 
    downsample if necessary, and save to a clean CSV.
    """
    dataset_name = dataset_enum.value
    clean_csv_path = os.path.join(paths['clean_dir'], f"{dataset_name}_svd{args.truncated}_n{args.max_sample}_clean.csv")
    
    if os.path.exists(clean_csv_path):
        logger.info(f"Clean CSV already exists at {clean_csv_path}. Skipping generation.")
        return clean_csv_path

    os.makedirs(paths['clean_dir'], exist_ok=True)

    # 1. Load Data via Auxiliary Function
    X_raw, y_raw = _load_raw_data(dataset_enum, paths, logger)

    # 2. Dynamic TruncatedSVD 
    # Safety Check: SVD will crash if n_features <= n_components
    n_features = X_raw.shape[1]
    if n_features > args.truncated:
        logger.info(f"Applying TruncatedSVD (reducing {n_features} to {args.truncated})...")
        svd = TruncatedSVD(n_components=args.truncated, random_state=42)
        X_dense = svd.fit_transform(X_raw)
        
        if 'svd_model_path' in paths:
            os.makedirs(os.path.dirname(paths['svd_model_path']), exist_ok=True)
            joblib.dump(svd, paths['svd_model_path'])
    else:
        logger.info(f"Dataset has {n_features} features, which is <= {args.truncated}. Skipping SVD.")
        X_dense = X_raw

    # 3. Downsampling
    if X_dense.shape[0] > args.max_sample:
        logger.info(f"Downsampling from {X_dense.shape[0]} to {args.max_sample}...")
        X_dense, y_raw = resample(
            X_dense, y_raw, 
            n_samples=args.max_sample, 
            stratify=y_raw, 
            random_state=42
        )

    # 4. Save to CSV
    logger.info("Saving to CSV...")
    col_names = [f"feature_{i}" for i in range(X_dense.shape[1])]
    df = pd.DataFrame(X_dense, columns=col_names)
    df['y'] = y_raw
    
    df.to_csv(clean_csv_path, index=False)
    logger.info(f"Saved cleanly reduced CSV to {clean_csv_path}")
    
    return clean_csv_path

def evaluate_clean_baseline(dataset_enum, paths, clean_csv_path, logger, aim_run):
    """
    Step 2: Train a clean classifier to get empirical accuracy, 
    and use the metalearner to get predicted accuracy on the test set.
    """
    dataset_name = dataset_enum.value
    logger.info(f"--- Step 2: Evaluating Clean Baseline for {dataset_name.upper()} ---")
    
    if not os.path.exists(paths['metalearner_path']):
        logger.error(f"MetaLearner not found at {paths['metalearner_path']}.")
        return

    # Load clean data
    logger.info("Loading clean data and splitting into Train/Test...")
    df = pd.read_csv(clean_csv_path)
    X = df.drop(columns=['y']).values
    y = df['y'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1. Empirical Accuracy
    logger.info("Training clean SVC to find empirical accuracy...")
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
    clf.fit(X_train, y_train)
    acc_emp = clf.score(X_test, y_test)

    # 2. Predicted Accuracy (via MetaLearner)
    logger.info("Extracting C-Measures on clean test set...")
    mfe = MFE(groups=["complexity"])
    mfe.fit(X_test, y_test)
    features, values = mfe.extract()
    
    X_meta_df = pd.DataFrame([dict(zip(features, values))]).fillna(0.0)
    
    logger.info("Predicting accuracy with MetaLearner...")
    meta_learner = joblib.load(paths['metalearner_path'])
    acc_pred = meta_learner.predict(X_meta_df.values)[0]

    # Print and Track
    logger.info("="*50)
    logger.info(f"RESULTS FOR: {dataset_name.upper()}")
    logger.info(f"Empirical Clean Accuracy: {acc_emp:.4f}")
    logger.info(f"Predicted Clean Accuracy: {acc_pred:.4f}")
    logger.info(f"Difference (Pred - Emp):  {acc_pred - acc_emp:.4f}")
    logger.info("="*50)

    # Log to Aim with dataset context
    aim_run.track(acc_emp, name='accuracy', context={'type': 'empirical_clean', 'dataset': dataset_name})
    aim_run.track(acc_pred, name='accuracy', context={'type': 'predicted_clean', 'dataset': dataset_name})
    aim_run.track(acc_pred - acc_emp, name='accuracy_difference', context={'dataset': dataset_name})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIVA Clean Baseline Evaluation for All Datasets")
    # Dataset argument removed; script automatically iterates all.
    parser.add_argument("--max_sample", type=int, default=50000, help="Downsample threshold")
    parser.add_argument("--metalearner", type=str, default="metalearner_clean_svr.pkl", help="Name of the metalearner model")
    parser.add_argument("--truncated", type=int, default=100, help="Feature reduction")
    parser.add_argument("--max_worker", type=int, default=None, help="Max worker used to extract in parallel complexity measure")
    parser.add_argument("--description", type=str, default="Evaluating clean baseline accuracy against Metalearner for all datasets", help="Description of the run for Aim")
    args = parser.parse_args()

    logger = setup_logger()

    run = Run(experiment="DIVA_BASELINE_ALL")
    run.set("description", args.description)
    run["hparams"] = vars(args)

    for dataset in DatasetEnum:
        try:
            logger.info(f"\n\n{'#'*60}\n### STARTING PIPELINE FOR: {dataset.value.upper()} ###\n{'#'*60}")
            
            # Construct dataset-specific paths
            base_folder = f"./data/test/{dataset.value}"
            paths = {
                'base_folder': base_folder,
                'npz_path': f"{base_folder}/{dataset.value}_processed_sparse.npz",
                'clean_dir': f"{base_folder}/clean_data",
                'svd_model_path': f"{base_folder}/{dataset.value}_svd_model.pkl",
                'metalearner_path': f"./data/{args.metalearner}"
            }

            logger.info(f"--- Step 1: Preparing Clean Data (SVD) for {dataset.value.upper()} ---")
            clean_file = step1_prepare_clean_data(dataset, args, paths, logger)
            
            evaluate_clean_baseline(dataset, paths, clean_file, logger, aim_run=run)
            
            logger.info(f"Successfully finished pipeline for {dataset.value.upper()}.")

        except Exception as e:
            # If one dataset fails, log the error but continue to the next dataset
            logger.exception(f"A fatal error occurred during pipeline execution for {dataset.value.upper()}:")

    logger.info("All Baseline Evaluations completed. Closing Aim run.")
    run.close()
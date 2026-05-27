import os
import random
import numpy as np
import pandas as pd
import glob
import logging
from enum import Enum
from tqdm import tqdm

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

# --- Modular Pipeline Imports ---
from scripts.data_generator.openml_fetcher import fetch_openml_datasets

# --- Restricted Poisoners for Regression Task ---
from scripts.poisoner.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.poisoner.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Regression_DataGenerator")

# ==========================================
# Configuration
# ==========================================
BASE_FOLDER = "data_2"
TARGETS_FILE = os.path.join(BASE_FOLDER, "regression_targets.csv")

POISONER_MAP = {
    "alfa_svm": AlfaPoisoner,
    "random_flip_svm": RandomFlipPoisoner,
}

RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ==========================================
# Evaluation Logic
# ==========================================
def get_average_accuracy(X_train, y_train, X_test, y_test):
    """Trains SVM, MLP, and RF, returning the average accuracy."""
    models = [
        SVC(kernel='rbf', random_state=42),
        MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
        RandomForestClassifier(n_estimators=100, random_state=42)
    ]
    
    accuracies = []
    for clf in models:
        try:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            accuracies.append(accuracy_score(y_test, preds))
        except Exception as e:
            logger.warning(f"Model {clf.__class__.__name__} failed to fit/predict: {e}")
            accuracies.append(0.0) # Penalize failure
            
    return np.mean(accuracies) if accuracies else 0.0

def _get_expected_poisoned_path(original_file, method, rate, poisoner_instance):
    """
    Constructs the exact output path by mimicking the logic inside the poisoner classes.
    """
    dataname = Path(original_file).stem
    
    # Handle the specific naming conventions used in your classes
    if method == "random_flip_svm":
        suffix = "randomlabelflip_svm"
    elif method == "alfa_svm":
        suffix = "alfa_svm"
    else:
        suffix = method # Fallback just in case
        
    # The files are saved with {rate:.2f}
    file_name = f"{dataname}_{suffix}_{rate:.2f}.csv"
    
    # Grab the exact output directory dictated by your BasePoisoner inheritance
    return os.path.join(poisoner_instance.poisoned_dir, file_name)

# ==========================================
# Synthetic Data Generator Logic
# ==========================================
def generate_synthetic_data(n_sets, folder):
    """Generates a diverse set of clean synthetic base datasets."""
    N_SAMPLES_OPTIONS = np.arange(100, 1500, 100) 
    data_path = os.path.join(folder, "clean_data")
    os.makedirs(data_path, exist_ok=True)
    feature_ranges = list(range(20, 121, 20)) 
    
    grid = [] 
    for f in feature_ranges:
        grid.append({"n_samples": N_SAMPLES_OPTIONS, "n_classes": [2], "n_features": [f], "n_informative": [int(f * 0.7), int(f * 0.8), int(f * 0.9)], "n_repeated": [0], "weights": [[0.5, 0.5], [0.55, 0.45]], "flip_y": [0.0, 0.01], "class_sep": [1.5, 2.0, 2.5]})
        grid.append({"n_samples": N_SAMPLES_OPTIONS, "n_classes": [2], "n_features": [f], "n_informative": [int(f * 0.4), int(f * 0.5), int(f * 0.6)], "n_repeated": [0], "weights": [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]], "flip_y": [0.03, 0.05, 0.08], "class_sep": [0.8, 1.0, 1.2]})

    param_sets = list(ParameterGrid(grid))
    selected_indices = np.random.choice(len(param_sets), n_sets, replace=(len(param_sets) < n_sets))
    generated_files = []

    for idx, i in enumerate(selected_indices):
        params = param_sets[i].copy()
        params["n_redundant"] = np.random.randint(0, max(1, params["n_features"] - params["n_informative"]))
        params["n_clusters_per_class"] = np.random.randint(1, 3) 
        params["random_state"] = np.random.randint(1000, 99999)

        X, y = make_classification(**params)
        X = StandardScaler().fit_transform(X)

        feature_names = [f"feature_{j}" for j in range(1, X.shape[1] + 1)]
        df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
        df["y"] = np.where(y > 0, 1, 0) 

        file_name = "f{:04d}_i{:03d}_n{:04d}_sep{:.1f}".format(params["n_features"], params["n_informative"], params["n_samples"], params["class_sep"])
        postfix = str(len(glob.glob(os.path.join(data_path, f"{file_name}_*.csv"))) + 1)
        output_path = os.path.join(data_path, f"{file_name}_{postfix}.csv")
        
        df.to_csv(output_path, index=False)
        generated_files.append(output_path)

    return generated_files

# ==========================================
# Orchestration Logic
# ==========================================
def process_dataset_for_regression(file_path, method):
    """
    Splits data, evaluates clean baseline, applies poison at all rates, 
    calculates drops, and returns data for the regressor if max_drop > 5%.
    """
    df = pd.read_csv(file_path)
    
    # Split into train/test to ensure standard evaluation 
    # We only poison the training set.
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_file_path = file_path.replace(".csv", "_train.csv")
    train_df.to_csv(train_file_path, index=False)
    
    X_train_clean, y_train_clean = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    
    # 1. Get Clean Baseline Accuracy
    clean_acc = get_average_accuracy(X_train_clean, y_train_clean, X_test, y_test)
    
    dataset_records = []
    max_drop = 0.0
    generated_poison_files = []
    
    poisoner = POISONER_MAP[method](base_folder=BASE_FOLDER)

    for rate in RATES:
        try:
            # Apply poisoner to the TRAINING set only
            poisoner.apply_poisoning(train_file_path, [rate])
            pois_file_path = _get_expected_poisoned_path(train_file_path, method, rate, poisoner)
            
            if not os.path.exists(pois_file_path):
                logger.warning(f"Expected poisoned file not found: {pois_file_path}")
                continue
                
            generated_poison_files.append(pois_file_path)
            pois_df = pd.read_csv(pois_file_path)
            X_train_pois, y_train_pois = pois_df.iloc[:, :-1], pois_df.iloc[:, -1]
            
            # 2. Get Poisoned Accuracy
            pois_acc = get_average_accuracy(X_train_pois, y_train_pois, X_test, y_test)
            
            # 3. Calculate Accuracy Drop
            acc_drop = clean_acc - pois_acc
            max_drop = max(max_drop, acc_drop)
            
            dataset_records.append({
                "original_clean_file": file_path,
                "poisoned_train_file": pois_file_path,
                "method": method,
                "requested_rate": rate,
                "clean_accuracy": clean_acc,
                "poisoned_accuracy": pois_acc,
                "accuracy_drop": acc_drop
            })
            
        except Exception as e:
            logger.error(f"Failed attacking {file_path} at rate {rate}: {e}")

    # Filtering Logic: If the maximum drop is <= 5%, discard everything.
    if max_drop <= 0.05:
        logger.info(f"Discarding {file_path} (Max Drop: {max_drop:.2%} <= 5%)")
        os.remove(file_path)
        os.remove(train_file_path)
        for pf in generated_poison_files:
            if os.path.exists(pf):
                os.remove(pf)
        return []
    
    logger.info(f"Keeping {file_path} (Max Drop: {max_drop:.2%} > 5%)")
    return dataset_records

def main():
    os.makedirs(BASE_FOLDER, exist_ok=True)
    
    # Initialize targets CSV
    if not os.path.exists(TARGETS_FILE):
        pd.DataFrame(columns=[
            "original_clean_file", "poisoned_train_file", "method", 
            "requested_rate", "clean_accuracy", "poisoned_accuracy", "accuracy_drop"
        ]).to_csv(TARGETS_FILE, index=False)

    n_synthetic = 250
    n_openml = 250
    
    logger.info(f"--- Generating {n_synthetic} Synthetic Tabular Datasets ---")
    synthetic_files = generate_synthetic_data(n_synthetic, BASE_FOLDER)
    
    logger.info(f"--- Fetching {n_openml} OpenML Tabular Datasets ---")
    openml_files = fetch_openml_datasets(
        n_max=n_openml, 
        folder=BASE_FOLDER, 
        max_retries=3, 
        db_path=None 
    )
    
    all_files = synthetic_files + openml_files
    
    valid_records = []
    
    for file in tqdm(all_files, desc="Evaluating and Poisoning Datasets"):
        # Randomly choose one poisoner per dataset
        method = random.choice(list(POISONER_MAP.keys()))
        
        records = process_dataset_for_regression(file, method)
        
        if records:
            valid_records.extend(records)
            # Iteratively save to avoid data loss on crash
            pd.DataFrame(records).to_csv(TARGETS_FILE, mode='a', header=False, index=False)

    logger.info("=====================================================")
    logger.info(f"✅ Pipeline Complete. Generated regression targets for {len(valid_records)} poisoned datasets.")
    logger.info(f"Targets saved to: {TARGETS_FILE}")
    logger.info("=====================================================")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()
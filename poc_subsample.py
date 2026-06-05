import os
import glob
import numpy as np
import pandas as pd
import openml
import joblib
import logging
import warnings
import random
import csv

from pymfe.mfe import MFE
from tabpfn import TabPFNClassifier

# --- Tabular Poisoner Imports ---
from scripts.poisoner.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.poisoner.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.poisoner.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.poisoner.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
from scripts.poisoner.svm_feature_collision.svm_featurecollision import FeatureCollisionPoisoner

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Top20SubsampleDetector")

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_PATH = "data/binary_detector.joblib"
OPTIMAL_THRESHOLD = 0.76  
MFE_GROUPS = ["complexity", "model-based", "landmarking"]
BASE_FOLDER = "data/temp"
CLEAN_DATA_DIR = os.path.join(BASE_FOLDER, "clean_data")
CHUNK_SIZE = 3000
RESULTS_CSV = "data/temp/evaluation_results.csv"

# Map string identifiers to their respective poisoner classes
POISONER_MAP = {
    "alfa_svm": AlfaPoisoner,
    "feature_noise_svm": FeatureNoisePoisoner,
    "random_flip_svm": RandomFlipPoisoner,
    "feature_collision": FeatureCollisionPoisoner,
    "pois_svm": PoisSVMPoisoner,
}
VALID_METHODS = list(POISONER_MAP.keys())

# ==============================================================================
# Pipeline Functions
# ==============================================================================
def get_top_20_large_datasets():
    """Fetches the top 20 datasets from OpenML with between 20,000 and 50,000 instances."""
    logger.info("Querying OpenML for datasets with 20,000 < instances < 50,000...")
    
    datasets_df = openml.datasets.list_datasets(output_format="dataframe")
    
    large_datasets = datasets_df[(datasets_df["NumberOfInstances"] > 20000) & 
                                 (datasets_df["NumberOfNumericFeatures"] < 100) & 
                                 (datasets_df["NumberOfInstances"] < 50000) & 
                                 (datasets_df["NumberOfNumericFeatures"] > 5)]
    top_20 = large_datasets.head(20)
    
    logger.info(f"Identified {len(top_20)} target datasets.")
    return top_20["did"].tolist()

def fetch_and_save_clean_dataset(did):
    """Fetches a specific OpenML dataset by ID, preprocesses it, and saves it."""
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    
    try:
        dataset = openml.datasets.get_dataset(did, download_data=True)
        target = dataset.default_target_attribute
        
        X, y, _, _ = dataset.get_data(target=target)
        
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        X_num = X_num.loc[:, X_num.std() > 0]
        if X_num.shape[1] == 0:
            logger.warning(f"Dataset {did} has no numeric features. Skipping.")
            return None
            
        y_binary = pd.factorize(y)[0]
        y_binary = np.where(y_binary == 0, 0, 1)
        
        df = pd.DataFrame(X_num.to_numpy(), columns=[f"feature_{i}" for i in range(X_num.shape[1])], dtype=np.float32)
        df["y"] = y_binary
        
        clean_path = os.path.join(CLEAN_DATA_DIR, f"openml_did_{did}_clean.csv")
        df.to_csv(clean_path, index=False)
        return clean_path
        
    except Exception as e:
        logger.error(f"Failed to process dataset {did}: {e}")
        return None

def apply_random_attack(clean_path, method, rate):
    """Applies a specific poisoner at a specific rate to the dataset and returns the resulting file path."""
    poisoner_cls = POISONER_MAP[method]
    poisoner = poisoner_cls(base_folder=BASE_FOLDER)
    
    metadata = poisoner.apply_poisoning(clean_path, [rate])
    return metadata[0]["Path"]

def extract_meta_features(X_chunk, y_chunk):
    """Extracts PyMFE features for a single chunk."""
    print(f"Extracting for {len(X_chunk)} points of {len(X_chunk[0])} features.")
    mfe = MFE(groups=MFE_GROUPS, random_state=42)
    mfe.fit(X_chunk, y_chunk)
    features, values = mfe.extract()
    return dict(zip(features, values))

def chunk_dataset_df(df, chunk_size=CHUNK_SIZE):
    """Subsamples a DataFrame into evenly sized chunks."""
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_chunks = len(df_shuffled) // chunk_size
    
    chunks = []
    for i in range(n_chunks):
        chunk = df_shuffled.iloc[i * chunk_size : (i + 1) * chunk_size]
        X_chunk = chunk.iloc[:, :-1].to_numpy()
        y_chunk = chunk.iloc[:, -1].to_numpy()
        chunks.append((X_chunk, y_chunk))
    return chunks

# ==============================================================================
# Execution and Accuracy Calculation
# ==============================================================================
def evaluate_strategy(chunks, clf):
    """Runs the chunk extraction and votes on the dataset's poison status."""
    if not chunks:
        return [], None
        
    meta_records = [extract_meta_features(X, y) for X, y in chunks]
    df_meta = pd.DataFrame(meta_records)

    df_meta = df_meta.fillna(0)
    df_meta = df_meta.replace([np.inf, -np.inf], 0)
    df_meta = df_meta.astype(np.float64)
    
    # Meta-Detection
    prob_poisoned = clf.predict_proba(df_meta)[:, 1]
    chunk_predictions = (prob_poisoned >= OPTIMAL_THRESHOLD).astype(int).tolist()
    
    poison_vote_ratio = np.mean(chunk_predictions)
    predicted_label = 1 if poison_vote_ratio > 0.50 else 0
    
    return chunk_predictions, predicted_label

def log_result_to_csv(filepath, rate, attack_type, num_points, chunk_preds, pred_label, true_label):
    """Appends a single evaluation result directly to the CSV."""
    is_correct = int(pred_label == true_label)
    row = [filepath, rate, attack_type, num_points, str(chunk_preds), pred_label, true_label, is_correct]
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    return is_correct

def main():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model missing at {MODEL_PATH}.")
        return

    # Prepare combinations for random attacks
    advx_range = np.round(np.arange(0.05, 0.31, 0.05), 2)
    all_possible_attacks = [(m, r) for m in VALID_METHODS for r in advx_range]

    # Initialize CSV header
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["path", "rate", "attack_type", "num_points", "chunk_predictions", "prediction_result", "true_label", "is_correct"])

    clf = joblib.load(MODEL_PATH)
    target_dids = get_top_20_large_datasets()
    
    correct_predictions = 0
    total_evaluations = 0

    # Ensure repeatability for random sampling
    np.random.seed(42)
    random.seed(42)

    for i, did in enumerate(target_dids):
        logger.info(f"\n--- Processing Dataset {i+1}/20 (DID: {did}) ---")
        
        # 1. Fetch Clean Data
        clean_file_path = fetch_and_save_clean_dataset(did)
        if not clean_file_path:
            continue
            
        df_clean = pd.read_csv(clean_file_path)
        num_points = len(df_clean)
        
        # 2. Evaluate Clean Set (True Label = 0)
        logger.info(f"Evaluating Clean Dataset (DID {did})...")
        clean_chunks = chunk_dataset_df(df_clean)
        chunk_preds, pred_label = evaluate_strategy(clean_chunks, clf)
        
        if pred_label is not None:
            is_correct = log_result_to_csv(clean_file_path, 0.0, "clean", num_points, chunk_preds, pred_label, 0)
            correct_predictions += is_correct
            total_evaluations += 1

        # 3. Apply 4 Random Attacks
        chosen_attacks = random.sample(all_possible_attacks, 4)
        for method, rate in chosen_attacks:
            try:
                poisoned_file_path = apply_random_attack(clean_file_path, method, rate)
            except Exception as e:
                logger.error(f"Poisoning failed for DID {did} ({method} @ {rate}): {e}")
                continue
            
            # Evaluate Poisoned Set (True Label = 1)
            logger.info(f"Evaluating Poisoned Dataset: {method} @ {rate} (DID {did})...")
            df_poisoned = pd.read_csv(poisoned_file_path)
            poisoned_chunks = chunk_dataset_df(df_poisoned)
            
            chunk_preds, pred_label = evaluate_strategy(poisoned_chunks, clf)
            
            if pred_label is not None:
                is_correct = log_result_to_csv(poisoned_file_path, rate, method, num_points, chunk_preds, pred_label, 1)
                correct_predictions += is_correct
                total_evaluations += 1

    # Print Final Accuracy
    if total_evaluations > 0:
        accuracy = correct_predictions / total_evaluations
        logger.info("\n========================================")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"Results saved to                : {RESULTS_CSV}")
        logger.info(f"Total Datasets Evaluated        : {total_evaluations}")
        logger.info(f"Correct Classifications         : {correct_predictions}")
        logger.info(f"Overall Accuracy                : {accuracy:.2%}")
        logger.info("========================================")
    else:
        logger.error("No evaluations were completed successfully.")

if __name__ == "__main__":
    main()
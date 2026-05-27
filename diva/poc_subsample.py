import os
import glob
import numpy as np
import pandas as pd
import openml
import joblib
import logging
import warnings
from pymfe.mfe import MFE
from tabpfn import TabPFNClassifier
from scripts.poisoner.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.poisoner.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.poisoner.svm_art.svm_art_generate_metadb import ArtSvmPoisoner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Top20SubsampleDetector")

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_PATH = "data/plots_loao_benchmark_all_features/loao_tabpfn_model_all_features.joblib"
OPTIMAL_THRESHOLD = 0.76  
MFE_GROUPS = ["complexity", "model-based", "landmarking"]
BASE_FOLDER = "temp"
CLEAN_DATA_DIR = os.path.join(BASE_FOLDER, "clean_data")
POISON_RATE = 0.20
CHUNK_SIZE = 3000

# ==============================================================================
# Pipeline Functions
# ==============================================================================
def get_top_20_large_datasets():
    """Fetches the top 20 datasets from OpenML with more than 30,000 instances."""
    logger.info("Querying OpenML for datasets with > 30,000 instances...")
    
    # Retrieve all datasets as a pandas dataframe 
    datasets_df = openml.datasets.list_datasets(output_format="dataframe")
    
    # Filter for > 30,000 instances and sort descending
    large_datasets = datasets_df[(datasets_df["NumberOfInstances"] > 20000) & (datasets_df["NumberOfNumericFeatures"]<100) & (datasets_df["NumberOfInstances"]<50000) & (datasets_df["NumberOfNumericFeatures"]>5)]
    top_20 = large_datasets.sort_values(by="NumberOfInstances", ascending=False).head(20)
    
    logger.info(f"Identified {len(top_20)} target datasets.")
    return top_20["did"].tolist()

def fetch_and_save_clean_dataset(did):
    """Fetches a specific OpenML dataset by ID, preprocesses it, and saves it."""
    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    
    try:
        dataset = openml.datasets.get_dataset(did, download_data=True)
        # Handle cases where default_target_attribute is missing
        target = dataset.default_target_attribute
        
        X, y, _, _ = dataset.get_data(target=target)
        
        # Keep numeric features only and fill missing values
        X_num = X.select_dtypes(include=[np.number]).fillna(0)
        if X_num.shape[1] == 0:
            logger.warning(f"Dataset {did} has no numeric features. Skipping.")
            return None
            
        # Ensure binary target mapping for PoisonFrogs
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

def attack_with_poison_frogs(clean_path):
    """Executes PoisSVM on the given dataset path."""
    poisoner = PoisSVMPoisoner(base_folder=BASE_FOLDER)
    poisoner.apply_poisoning(clean_path, [POISON_RATE])
    
    base_name = os.path.splitext(os.path.basename(clean_path))[0]
    search_pattern = os.path.join(BASE_FOLDER, "**", f"*{base_name}*poissvm_svm*.csv")
    poisoned_files = glob.glob(search_pattern, recursive=True)
    
    if not poisoned_files:
        raise FileNotFoundError(f"Failed to locate PoisSVM output for {base_name}.")
        
    poisoned_files.sort(key=os.path.getmtime)
    return poisoned_files[-1]

def attack_with_randomflip(clean_path):
    """Executes RandomLabel on the given dataset path."""
    poisoner = RandomFlipPoisoner(base_folder=BASE_FOLDER)
    poisoner.apply_poisoning(clean_path, [POISON_RATE])
    
    base_name = os.path.splitext(os.path.basename(clean_path))[0]
    search_pattern = os.path.join(BASE_FOLDER, "**", f"*{base_name}*randomlabel*.csv")
    poisoned_files = glob.glob(search_pattern, recursive=True)
    
    if not poisoned_files:
        raise FileNotFoundError(f"Failed to locate RandomLabel output for {base_name}.")
        
    poisoned_files.sort(key=os.path.getmtime)
    return poisoned_files[-1]

def attack_with_art(clean_path):
    """Executes ArtSVM on the given dataset path."""
    poisoner = ArtSvmPoisoner(base_folder=BASE_FOLDER)
    poisoner.apply_poisoning(clean_path, [POISON_RATE])
    
    base_name = os.path.splitext(os.path.basename(clean_path))[0]
    search_pattern = os.path.join(BASE_FOLDER, "**", f"*{base_name}*art_svm*.csv")
    poisoned_files = glob.glob(search_pattern, recursive=True)
    
    if not poisoned_files:
        raise FileNotFoundError(f"Failed to locate Art output for {base_name}.")
        
    poisoned_files.sort(key=os.path.getmtime)
    return poisoned_files[-1]

def extract_meta_features(X_chunk, y_chunk):
    """Extracts PyMFE features for a single chunk."""
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
def evaluate_strategy(chunks, clf, true_label):
    """Runs the chunk extraction and votes on the dataset's poison status."""
    if not chunks:
        return None
        
    meta_records = [extract_meta_features(X, y) for X, y in chunks]
    df_meta = pd.DataFrame(meta_records)

    df_meta = df_meta.fillna(0)
    # 2. Replace infinity with 0 (or a very large finite number)
    df_meta = df_meta.replace([np.inf, -np.inf], 0)
    # 3. Ensure float64
    df_meta = df_meta.astype(np.float64)
    
    # Meta-Detection
    prob_poisoned = clf.predict_proba(df_meta)[:, 1]
    chunk_predictions = (prob_poisoned >= OPTIMAL_THRESHOLD).astype(int)
    
    poison_vote_ratio = np.mean(chunk_predictions)
    predicted_label = 1 if poison_vote_ratio > 0.50 else 0
    
    return predicted_label == true_label

def main():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model missing at {MODEL_PATH}.")
        return

    clf = joblib.load(MODEL_PATH)
    target_dids = get_top_20_large_datasets()
    
    correct_predictions = 0
    total_evaluations = 0

    for i, did in enumerate(target_dids):
        logger.info(f"\n--- Processing Dataset {i+1}/20 (DID: {did}) ---")
        
        # 1. Fetch Clean Data
        clean_file_path = fetch_and_save_clean_dataset(did)
        if not clean_file_path:
            continue
            
        # 2. Poison Data
        try:
            poisoned_file_path = attack_with_randomflip(clean_file_path)
        except Exception as e:
            logger.error(f"Poisoning failed for DID {did}: {e}")
            continue
            
        df_clean = pd.read_csv(clean_file_path)
        df_poisoned = pd.read_csv(poisoned_file_path)
        
        # 3. Chunk Data
        clean_chunks = chunk_dataset_df(df_clean)
        poisoned_chunks = chunk_dataset_df(df_poisoned)
        
        # 4. Evaluate Clean Set (True Label = 0)
        logger.info(f"Evaluating Clean Parent Dataset (DID {did})...")
        if evaluate_strategy(clean_chunks, clf, true_label=0):
            correct_predictions += 1
        total_evaluations += 1
        
        # 5. Evaluate Poisoned Set (True Label = 1)
        logger.info(f"Evaluating Poisoned Parent Dataset (DID {did})...")
        if evaluate_strategy(poisoned_chunks, clf, true_label=1):
            correct_predictions += 1
        total_evaluations += 1

    # Print Final Accuracy
    if total_evaluations > 0:
        accuracy = correct_predictions / total_evaluations
        logger.info("\n========================================")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"Total Parent Datasets Evaluated : {total_evaluations}")
        logger.info(f"Correct Classifications         : {correct_predictions}")
        logger.info(f"Overall Accuracy                : {accuracy:.2%}")
        logger.info("========================================")
    else:
        logger.error("No evaluations were completed successfully.")

if __name__ == "__main__":
    main()
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
    logger.info("--- Step 4: DIVA Evaluation ---")
    meta_db_path = os.path.join(paths['meta_db'])
    
    if not os.path.exists(meta_db_path):
        logger.error(f"MetaDB not found at {meta_db_path}. Ensure the poisoning pipeline ran successfully.")
        return

    logger.info(f"Loading Meta-Learner from {paths['metalearner_path']}")
    meta_learner = joblib.load(paths['metalearner_path'])
    
    df_meta = pd.read_csv(meta_db_path).sort_values(by="Rate")
    results_emp, results_pred, results_flags, rates = [], [], [], []

    exclude_cols = ['Data', 'Path.Poison', 'Rate', 'Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison', 'key', 'file', 'error']
    feature_cols = [c for c in df_meta.columns if c not in exclude_cols]

    for _, row in df_meta.iterrows():
        rate = row['Rate']
        acc_emp = row['Test.Poison'] 
        
        X_meta_df = pd.DataFrame([row[feature_cols]])
        X_meta = X_meta_df.fillna(0.0).values
        
        acc_pred = meta_learner.predict(X_meta)[0]
        is_flagged = acc_pred - acc_emp > (acc_emp * 0.05)
        
        rates.append(rate)
        results_emp.append(acc_emp)
        results_pred.append(acc_pred)
        results_flags.append(is_flagged)
        
        logger.info(f"Rate {rate:.2f} | Emp: {acc_emp:.4f} | Pred: {acc_pred:.4f} | Flag: {is_flagged}")

        step_idx = int(rate * 100) 
        aim_run.track(acc_emp, name='accuracy', context={'type': 'empirical'}, step=step_idx)
        aim_run.track(acc_pred, name='accuracy', context={'type': 'predicted'}, step=step_idx)
        aim_run.track(float(is_flagged), name='diva_flag_triggered', step=step_idx)
        aim_run.track(rate, name='poisoning_rate', step=step_idx)

    generate_diva_plot(args, rates, results_emp, results_pred, results_flags, logger, aim_run)

def generate_diva_plot(args, rates, empirical, predicted, flags, logger, aim_run):
    logger.info("Generating DIVA Detection Plot...")
    fig = plt.figure(figsize=(10, 6))
    plt.plot(rates, empirical, 'o-', label='Empirical Acc (Poisoned)')
    plt.plot(rates, predicted, 's--', label='Predicted Acc (Clean Expectation)')
    
    for i, flag in enumerate(flags):
        if flag:
            plt.axvspan(rates[i]-0.02, rates[i]+0.02, color='red', alpha=0.1, label='DIVA Flag' if i==0 else "")
            
    plt.title(f"DIVA Detection: Empirical vs Predicted Accuracy ({args.method.upper()})")
    plt.xlabel("Poisoning Rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)

    aim_run.track(Image(fig), name='diva_detection_plot', context={'dataset': args.dataset})
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIVA Poisoning and Evaluation Pipeline")
    # Dynamically pull choices from the Enum
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
        'metalearner_path': f"./data/metalearners/{args.metalearner}"
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
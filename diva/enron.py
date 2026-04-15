import os
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
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

# --- Mapping methods to their respective classes ---
POISONER_MAP = {
    "poissvm": PoisSVMPoisoner,
    "feature_noise": FeatureNoisePoisoner,
    "random_flip": RandomFlipPoisoner,
    "alfa": AlfaPoisoner,
    "falfa": FalfaNNPoisoner,
    "art": ArtSvmPoisoner
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
    Step 1: Load NPZ, universally apply TruncatedSVD(100), and save to a clean CSV.
    """
    clean_csv_path = os.path.join(paths['clean_dir'], f"{args.dataset}_svd100_clean.csv")
    os.makedirs(paths['clean_dir'], exist_ok=True)
    
    # If the clean SVD file already exists, we can skip re-computing
    if os.path.exists(clean_csv_path):
        logger.info(f"Clean SVD dataset already exists at {clean_csv_path}. Skipping SVD computation.")
        return clean_csv_path

    logger.info(f"Loading raw data from {paths['npz_path']}")
    f = np.load(paths['npz_path'], allow_pickle=True, encoding='latin1')
    X_train_sparse = f['X_train'].reshape(1)[0]
    
    # Map labels to 0 and 1 for safe processing across all methods
    y_train = np.where(f['Y_train'] > 0, 1, 0)

    logger.info("Applying universal TruncatedSVD (n_components=100)...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train_dense = svd.fit_transform(X_train_sparse)
    
    # Save SVD model for future test-set transformations if needed
    os.makedirs(os.path.dirname(paths['svd_model_path']), exist_ok=True)
    joblib.dump(svd, paths['svd_model_path'])

    # Optional Downsampling to prevent memory exhaustion in ART/ALFA
    if X_train_dense.shape[0] > args.max_sample:
        logger.info(f"Downsampling from {X_train_dense.shape[0]} to {args.max_sample}...")
        X_train_dense, y_train = resample(X_train_dense, y_train, n_samples=args.max_sample, stratify=y_train, random_state=42)

    # Save to CSV for the BasePoisoner to consume
    df = pd.DataFrame(X_train_dense)
    df['y'] = y_train
    df.to_csv(clean_csv_path, index=False)
    logger.info(f"Saved cleanly reduced CSV to {clean_csv_path}")
    
    return clean_csv_path


def step4_evaluate_diva(args, paths, logger, aim_run):
    """
    Step 4 & 5: Load the resulting MetaDB, predict via Metalearner, and track via Aim.
    """
    logger.info("--- Step 4: DIVA Evaluation ---")
    meta_db_path = os.path.join(paths['meta_db'])
    
    if not os.path.exists(meta_db_path):
        logger.error(f"MetaDB not found at {meta_db_path}. Ensure the poisoning pipeline ran successfully.")
        return

    logger.info(f"Loading Meta-Learner from {paths['metalearner_path']}")
    meta_learner = joblib.load(paths['metalearner_path'])
    
    df_meta = pd.read_csv(meta_db_path).sort_values(by="Rate")
    
    results_emp, results_pred, results_flags, rates = [], [], [], []

    # Filter out non-complexity columns to feed the metalearner
    exclude_cols = ['Data', 'Path.Poison', 'Rate', 'Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison', 'key', 'file', 'error']
    feature_cols = [c for c in df_meta.columns if c not in exclude_cols]

    for _, row in df_meta.iterrows():
        rate = row['Rate']
        acc_emp = row['Test.Poison'] # Empirical poisoned accuracy 
        
        # Meta-feature vector
        X_meta_df = pd.DataFrame([row[feature_cols]])
        X_meta = X_meta_df.fillna(0.0).values
        
        # DIVA Prediction
        acc_pred = meta_learner.predict(X_meta)[0]
        is_flagged = abs(acc_emp - acc_pred) > (acc_emp * 0.05)
        
        rates.append(rate)
        results_emp.append(acc_emp)
        results_pred.append(acc_pred)
        results_flags.append(is_flagged)
        
        logger.info(f"Rate {rate:.2f} | Emp: {acc_emp:.4f} | Pred: {acc_pred:.4f} | Flag: {is_flagged}")

        # Step 5: Save metrics to Aim
        step_idx = int(rate * 100) 
        aim_run.track(acc_emp, name='accuracy', context={'type': 'empirical'}, step=step_idx)
        aim_run.track(acc_pred, name='accuracy', context={'type': 'predicted'}, step=step_idx)
        aim_run.track(float(is_flagged), name='diva_flag_triggered', step=step_idx)
        aim_run.track(rate, name='poisoning_rate', step=step_idx)

    # Plotting
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

    # Log plot to aim
    aim_run.track(Image(fig), name='diva_detection_plot', context={'dataset': args.dataset})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIVA Poisoning and Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., enron, imdb)")
    parser.add_argument("--method", type=str, required=True, choices=POISONER_MAP.keys(), help="Poisoning method to apply")
    parser.add_argument("--max", type=float, default=0.41, help="Max poisoning rate")
    parser.add_argument("--step", type=float, default=0.05, help="Poisoning rate step size")
    parser.add_argument("--max_sample", type=int, default=50000, help="Downsample threshold")
    parser.add_argument("--metalearner", type=str, default="metalearner_feature_noise_svm+random_flip_svm+poissvm_svm+alfa_svm.pkl", help="Name of the metalearner model")
    parser.add_argument("--description", type=str, default="", help="Description of the run for Aim")
    args = parser.parse_args()

    # --- Setup Paths ---
    base_folder = f"./data/test/{args.dataset}"
    paths = {
        'base_folder': base_folder,
        'npz_path': f"{base_folder}/{args.dataset}_processed_sparse.npz",
        'clean_dir': f"{base_folder}/clean_data",
        'svd_model_path': f"{base_folder}/{args.dataset}_svd_model.pkl",
        'metalearner_path': f"./data/metalearners/{args.metalearner}"
    }

    logger = setup_logger(args.dataset, args.method)
    
    # Initialize Aim Run
    run = Run(experiment=f"DIVA_{args.dataset.upper()}")
    run["hparams"] = vars(args)

    try:
        # Step 1: Load and reduce dataset (TruncatedSVD)
        logger.info("--- Step 1: Preparing Clean Data (SVD) ---")
        clean_file = step1_prepare_clean_data(args, paths, logger)
        
        # Step 2 & 3: Run BasePoisoner Pipeline (Poison -> C-Measure -> MetaDB)
        logger.info("--- Step 2 & 3: Executing Poisoning and C-Measure Extraction ---")
        advx_range = np.arange(0.0, args.max, args.step)
        
        # Instantiate the correct poisoner dynamically
        PoisonerClass = POISONER_MAP[args.method]
        poisoner = PoisonerClass(base_folder=paths['base_folder'])
        
        # Entrypoint="poison" runs the full suite automatically in BasePoisoner
        meta_db = poisoner.run_pipeline([clean_file], advx_range, entrypoint="poison")
        paths['meta_db'] = meta_db
        
        # Step 4 & 5: Evaluate predictions and track with Aim
        step4_evaluate_diva(args, paths, logger, aim_run=run)
        
        logger.info("DIVA Evaluation Pipeline completed successfully.")
    except Exception as e:
        logger.exception("A fatal error occurred during pipeline execution:")
    finally:
        run.close()
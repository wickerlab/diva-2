import os
import argparse
import random
import numpy as np
import pandas as pd
import logging
import concurrent.futures
import re
import joblib
import glob
from aim import Run
from pathlib import Path
from tqdm import tqdm
from pymfe.mfe import MFE

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

# --- Import your Specific Poisoners ---
from scripts.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
from scripts.svm_art.svm_art_generate_metadb import ArtSvmPoisoner
from scripts.svm_biggio.svm_biggio_generate_metadb import BiggioSvmPoisoner

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Pipeline_Orchestrator")

POISONER_MAP = {
    "alfa_svm": AlfaPoisoner,
    "feature_noise_svm": FeatureNoisePoisoner,
    "random_flip_svm": RandomFlipPoisoner,
    "poissvm_svm": PoisSVMPoisoner,
    "biggio_svm": BiggioSvmPoisoner,
    "art_svm": ArtSvmPoisoner
}

# Define the method pools based on dataset constraints
FAST_METHODS = ["alfa_svm", "feature_noise_svm", "random_flip_svm", "poissvm_svm"]
ALL_METHODS = list(POISONER_MAP.keys())

def generate_synthetic_data(n_sets, folder):
    """Generates a diverse set of clean synthetic base datasets."""
    N_SAMPLES_OPTIONS = np.arange(100, 1501, 100) 
    data_path = os.path.join(folder, "clean_data")
    os.makedirs(data_path, exist_ok=True)
    feature_ranges = list(range(20, 121, 20)) 
    
    grid = [] 
    for f in feature_ranges:
        grid.append({"n_samples": N_SAMPLES_OPTIONS, "n_classes": [2], "n_features": [f], "n_informative": [int(f * 0.7), int(f * 0.8), int(f * 0.9)], "n_repeated": [0], "weights": [[0.5, 0.5], [0.55, 0.45]], "flip_y": [0.0, 0.01], "class_sep": [1.5, 2.0, 2.5]})
        grid.append({"n_samples": N_SAMPLES_OPTIONS, "n_classes": [2], "n_features": [f], "n_informative": [int(f * 0.4), int(f * 0.5), int(f * 0.6)], "n_repeated": [0], "weights": [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]], "flip_y": [0.03, 0.05, 0.08], "class_sep": [0.8, 1.0, 1.2]})
        grid.append({"n_samples": N_SAMPLES_OPTIONS, "n_classes": [2], "n_features": [f], "n_informative": [int(f * 0.2), int(f * 0.3)], "n_repeated": [0], "weights": [[0.5, 0.5], [0.7, 0.3], [0.8, 0.2]], "flip_y": [0.10, 0.15, 0.20], "class_sep": [0.3, 0.5, 0.7]})

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

        file_name = "f{:04d}_i{:03d}_r{:03d}_n{:04d}_sep{:.1f}".format(params["n_features"], params["n_informative"], params["n_redundant"], params["n_samples"], params["class_sep"])

        postfix = str(len(glob.glob(os.path.join(data_path, f"{file_name}_*.csv"))) + 1)
        output_path = os.path.join(data_path, f"{file_name}_{postfix}.csv")
        df.to_csv(output_path, index=False)
        generated_files.append(output_path)

    return generated_files

def _extract_cmeasures_standalone(file_path):
    """Standalone parallel-friendly MFE extraction strictly using file paths."""
    try:
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        y = np.where(y == -1, 0, y)

        mfe = MFE(groups=["complexity"])
        mfe.fit(X, y)
        features, values = mfe.extract()

        result = {"Path": file_path}
        result.update(dict(zip(features, values)))
        return result
    except Exception as e:
        return {"Path": file_path, "error": str(e)}

def prepare_meta_dataset(df):
    drop_cols = ['Data', 'Path', 'Method', 'Rate', 'Is_Poisoned', 'error']
    drop_cols += [c for c in df.columns if c in ['Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison', 'key', 'file']]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    df[feature_cols] = df[feature_cols].fillna(0)
    return df, feature_cols

def train_and_evaluate(df, feature_cols, run):
    X = df[feature_cols]
    y = df['Is_Poisoned']
    groups = df['Data'] 
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    logger.info(f"Training Meta-Classifier on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }
    
    for metric_name, value in metrics.items():
        run.track(value, name=metric_name, context={"subset": "test"})
        logger.info(f"  {metric_name.capitalize()}: {value:.4f}")
        
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=False).head(10)
    logger.info("\nTop 10 Universal Predictive C-Measures:")
    for _, row in feat_imp.iterrows():
        logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
        run.track(row['Importance'], name=f"importance_{row['Feature']}")
        
    return metrics, clf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nSets", default=50, type=int, help="Number of synthetic datasets to generate/use.")
    parser.add_argument("-a", "--n_attacks", default=6, type=int, help="Number of random attacks to perform PER dataset.")
    parser.add_argument("-f", "--folder", default="data", type=str, help="Base output folder.")
    parser.add_argument("-s", "--step", type=float, default=0.05, help="Poisoning rate step interval.")
    parser.add_argument("-m", "--max", type=float, default=0.41, help="Max poisoning rate.")
    parser.add_argument("-w", "--workers", type=int, default=None, help="Max workers for MFE extraction.")
    args = parser.parse_args()

    base = args.folder
    random.seed(42)
    advx_range = np.arange(args.step, args.max, args.step) # Exclude 0.0 from generation loops

    run = Run(experiment="Optimized-CMeasure-Pipeline")
    run["hparams"] = vars(args)

    # =========================================================================
    # STEP 1: GENERATE DATA
    # =========================================================================
    logger.info("\n=== STEP 1: Gathering Clean Datasets ===")
    if args.nSets > 0:
        clean_files = generate_synthetic_data(args.nSets, args.folder)
    else:
        clean_files = glob.glob(f"{args.folder}/clean_data/*.csv")
    
    logger.info(f"Loaded {len(clean_files)} clean datasets.")

    # =========================================================================
    # STEP 2: CREATE RANDOM ATTACK PLAN
    # =========================================================================
    logger.info("\n=== STEP 2: Creating Random Attack Plan ===")
    attack_plan = {m: {} for m in ALL_METHODS}
    metadata_records = []

    for file in clean_files:
        dataname = Path(file).stem
        
        # 1. Register the Clean file in metadata
        metadata_records.append({
            "Data": dataname, "Path": file, "Method": "clean", "Rate": 0.0, "Is_Poisoned": 0
        })

        # 2. Decide available methods based on size
        match = re.search(r'_n(\d+)_', dataname)
        n_samples = int(match.group(1)) if match else 500
        pool = ALL_METHODS if n_samples < 500 else FAST_METHODS

        # 3. Pick N random unique combinations of (method, rate)
        all_combos = [(m, r) for m in pool for r in advx_range]
        n_picks = min(args.n_attacks, len(all_combos))
        chosen_attacks = random.sample(all_combos, n_picks)

        for m, r in chosen_attacks:
            attack_plan[m].setdefault(file, []).append(r)

    # =========================================================================
    # STEP 3: EXECUTE POISONING SELECTIVELY
    # =========================================================================
    logger.info("\n=== STEP 3: Executing Poisoning ===")
    for method, file_rate_map in attack_plan.items():
        if not file_rate_map: continue
        
        logger.info(f"--- Running {method.upper()} on {len(file_rate_map)} files ---")
        poisoner = POISONER_MAP[method](base_folder=base)
        
        for file, rates in tqdm(file_rate_map.items(), desc=f"{method}", ncols=100):
            # Sort rates so cumulative attacks (Art, Biggio, PoisSVM) work correctly
            sorted_rates = sorted(rates)
            
            # The poisoner will only calculate the specific rates passed to it
            # It will utilize the surrogate state caching perfectly for these specific rates
            poisoner.apply_poisoning(file, sorted_rates)

    # =========================================================================
    # STEP 4: HARVEST METADATA & BULK C-MEASURE EXTRACTION
    # =========================================================================
    logger.info("\n=== STEP 4: Bulk C-Measure Extraction ===")
    
    # Read the individual csv_score logs from the poisoners to find all generated file paths
    for method in ALL_METHODS:
        score_file = os.path.join(base, "poisoned_data", f"synth_{method}_score.csv")
        if os.path.exists(score_file):
            df = pd.read_csv(score_file)
            for _, row in df.iterrows():
                path = row['Path.Poison']
                if os.path.exists(path):
                    metadata_records.append({
                        "Data": row['Data'], "Path": path, "Method": method, 
                        "Rate": row['Rate'], "Is_Poisoned": 1 if row['Rate'] > 0 else 0
                    })

    # Deduplicate metadata (in case poisoners logged duplicate rows)
    meta_df = pd.DataFrame(metadata_records).drop_duplicates(subset=["Path"])
    meta_df.to_csv(os.path.join(base, "attack_metadata.csv"), index=False)
    
    files_to_compute = meta_df['Path'].tolist()
    cmeasure_results = []
    
    logger.info(f"Extracting C-Measures for {len(files_to_compute)} total files (Clean + Poisoned)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_extract_cmeasures_standalone, f): f for f in files_to_compute}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files_to_compute), desc="pymfe"):
            cmeasure_results.append(future.result())
            
    cmeasure_df = pd.DataFrame(cmeasure_results)
    
    # Merge Metadata with C-Measures into ONE master file
    final_db = pd.merge(meta_df, cmeasure_df, on="Path", how="inner")
    final_db_path = os.path.join(base, "universal_meta_database.csv")
    final_db.to_csv(final_db_path, index=False)
    logger.info(f"Saved Unified Database to: {final_db_path}")

    # =========================================================================
    # STEP 5: TRAIN META-CLASSIFIER
    # =========================================================================
    logger.info("\n=== STEP 5: Training Meta-Classifier ===")
    df_prepared, feature_cols = prepare_meta_dataset(final_db)
    metrics, final_model = train_and_evaluate(df_prepared, feature_cols, run)

    model_save_path = os.path.join(base, "universal_meta_classifier.joblib")
    joblib.dump(final_model, model_save_path)
    logger.info(f"\n✅ Optimization Complete! Model saved to: {model_save_path}")

    run.close()
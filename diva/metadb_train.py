import os
import argparse
import random
import numpy as np
import pandas as pd
import logging
import joblib
import glob
from pathlib import Path
from tqdm import tqdm
import openml
import aim
from aim.sdk.reporter import RunStatusReporter
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# --- Modular Pipeline Imports ---
from scripts.cmeasures import compute_cmeasures
from scripts.meta_db import append_to_db, sync_filesystem_to_metadb

# --- Import your Specific Poisoners ---
from scripts.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
from scripts.svm_art.svm_art_generate_metadb import ArtSvmPoisoner
from scripts.svm_biggio.svm_biggio_generate_metadb import BiggioSvmPoisoner
from scripts.svm_feature_collision.svm_featurecollision import FeatureCollisionPoisoner

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DIVA_Training_Orchestrator")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s'))
    logger.addHandler(ch)

POISONER_MAP = {
    "alfa_svm": AlfaPoisoner,
    "feature_noise_svm": FeatureNoisePoisoner,
    "random_flip_svm": RandomFlipPoisoner,
    #"poissvm_svm": PoisSVMPoisoner,
    "biggio_svm": BiggioSvmPoisoner,
    #"art_svm": ArtSvmPoisoner,
    "feature_collision": FeatureCollisionPoisoner
}

def plot_feature_importances(importances, feature_names, model_name, save_path):
    """Generates and saves a bar plot of the top 20 C-Measure feature importances."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 8))
    
    indices = np.argsort(importances)[::-1][:20]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title(f"Top 20 C-Measure Importances ({model_name})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Importance Score", fontsize=12, fontweight='bold')
    plt.ylabel("C-Measure", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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



def fetch_openml_datasets(n_max, folder, max_retries=3, db_path=None):
    """Fetches real-world binary classification datasets from OpenML, skipping existing ones."""
    logger.info(f"Fetching up to {n_max} OpenML binary datasets...")
    data_path = os.path.join(folder, "clean_data")
    os.makedirs(data_path, exist_ok=True)
    
    # --- ADDED: Read metadatabase to completely ignore already processed files ---
    processed_paths = set()
    if db_path and os.path.exists(db_path):
        try:
            processed_paths = set(pd.read_csv(db_path, usecols=['Path'])['Path'].values)
        except Exception:
            pass
    # ---------------------------------------------------------------------------
    
    # --- ADDED: Retry mechanism for the OpenML API ---
    datasets_df = None
    for attempt in range(max_retries):
        try:
            datasets_df = openml.datasets.list_datasets(output_format='dataframe')
            break # Success, break out of retry loop
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            logger.warning(f"OpenML server error on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s... ({e})")
            time.sleep(wait_time)
            
    if datasets_df is None:
        logger.error("Failed to connect to OpenML after multiple attempts. Aborting OpenML fetch.")
        return []
    # --------------------------------------------------

    binary_datasets = datasets_df[
        (datasets_df['NumberOfClasses'] == 2) & 
        (datasets_df['NumberOfInstances'] >= 500) &
        (datasets_df['NumberOfInstances'] <= 2000) & 
        (datasets_df['NumberOfMissingValues'] == 0) &
        (datasets_df['NumberOfNumericFeatures'] > 5) &
        (datasets_df['NumberOfNumericFeatures'] < 6000)
    ]
    
    generated_files = []
    count = 0
    
    for row in binary_datasets.itertuples():
        if count >= n_max: 
            break
            
        did = row.did
        dataset_name = row.name
        safe_name = str(dataset_name).lower().replace(' ', '_').replace('/', '')
        
        existing_files = glob.glob(os.path.join(data_path, f"openml_{safe_name}_*.csv"))
        if existing_files:
            local_file = existing_files[0]
            
            # 1. If it is already fully processed in the metadatabase, ignore it entirely
            if local_file in processed_paths:
                continue 
                
            # 2. If it is downloaded locally but NOT in the metadatabase, reuse it!
            generated_files.append(local_file)
            count += 1
            logger.info(f"Dataset '{dataset_name}' already exists locally. Skipping download. ({count}/{n_max})")
            continue
            
        # --- ADDED: Inner retry mechanism for individual downloads ---
        for attempt in range(max_retries):
            try:
                dataset = openml.datasets.get_dataset(did)
                X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
                X_num = X.select_dtypes(include=['number']).dropna(axis=1)
                
                if X_num.shape[1] < 5: 
                    break # Not enough features, move to next dataset
                
                X_scaled = StandardScaler().fit_transform(X_num)
                y_binary = pd.factorize(y)[0]
                
                df = pd.DataFrame(X_scaled, columns=[f"feature_{i}" for i in range(X_scaled.shape[1])], dtype=np.float32)
                df["y"] = y_binary
                
                file_name = f"openml_{safe_name}_n{len(y_binary)}_f{X_scaled.shape[1]}.csv"
                output_path = os.path.join(data_path, file_name)
                
                df.to_csv(output_path, index=False)
                generated_files.append(output_path)
                count += 1
                logger.info(f"Successfully loaded OpenML dataset '{dataset.name}' ({count}/{n_max})")
                break # Success, break out of retry loop
                
            except Exception as e:
                if "107" in str(e) or "server load" in str(e).lower():
                    logger.warning(f"OpenML server busy while downloading '{dataset_name}'. Retrying...")
                    time.sleep(3)
                else:
                    break # Not a server timeout error, just a weird dataset. Move on.
            
    return generated_files

def augment_training_db(db_path, base_folder, n_datasets, n_attacks, workers, source, methods):
    """Orchestrates creating new data and adding it to the Universal Training DB"""
    if source == "synthetic":
        new_clean_files = generate_synthetic_data(n_datasets, base_folder)
    elif source == "openml":
        new_clean_files = fetch_openml_datasets(n_datasets, base_folder, db_path=db_path)

    advx_range = np.round(np.arange(0.1, 0.41, 0.05), 2)
    
    # Create a pool of ALL possible (method, rate) tuples
    all_possible_attacks = [
        (m, r) for m in POISONER_MAP.keys() 
        if (methods is None or m in methods) 
        for r in advx_range
    ]
    
    all_generated_metadata = []

    for file in new_clean_files:
        dataname = Path(file).stem
        all_generated_metadata.append({"Data": dataname, "Path": file, "Method": "clean", "Rate": 0.0, "Is_Poisoned": 0})
        
        # Sample exactly n_attacks combinations from the global pool
        chosen_attacks = random.sample(all_possible_attacks, min(n_attacks, len(all_possible_attacks)))

        logger.info(f"Chosen attacks: {chosen_attacks}")
        
        # Group the randomly chosen combinations by method 
        # so we can pass the specific list of rates to the poisoner's apply_poisoning function
        attack_plan = {}
        for method, rate in chosen_attacks:
            attack_plan.setdefault(method, []).append(rate)
            
        # Execute the specific rates for each method
        for method, rates in attack_plan.items():
            poisoner = POISONER_MAP[method](base_folder=base_folder)
            
            # Pass only the specifically sampled rates, sorted chronologically 
            generated_meta = poisoner.apply_poisoning(file, sorted(rates))
            if generated_meta:
                all_generated_metadata.extend(generated_meta)

    paths_to_compute = [m["Path"] for m in all_generated_metadata]
    
    # Compute using the db_path to instantly skip files already processed
    cmeasures_df = compute_cmeasures(paths_to_compute, workers=workers, db_path=db_path)
    append_to_db(db_path, all_generated_metadata, cmeasures_df)

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'Poisoned'], yticklabels=['Clean', 'Poisoned'])
    plt.title(f"Confusion Matrix ({model_name})", fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_method_accuracy(y_true, y_pred, methods, model_name, save_path):
    results = []
    for method in np.unique(methods):
        mask = (methods == method)
        acc = accuracy_score(y_true[mask], y_pred[mask])
        results.append({'Method': method.upper(), 'Accuracy': acc})
    
    df_res = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x='Accuracy', y='Method', palette='viridis')
    plt.title(f"Detection Accuracy by Attack Method ({model_name})", fontweight='bold')
    plt.xlim(0, 1.05)
    for index, value in enumerate(df_res['Accuracy']):
        plt.text(value + 0.01, index, f"{value:.1%}", va='center')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_heatmap(y_true, y_pred, methods, rates, model_name, save_path):
    df_eval = pd.DataFrame({'Method': methods, 'Rate': rates, 'Correct': (y_true == y_pred).astype(int)})
    
    # Clean up names for the plot
    df_eval['Method'] = df_eval['Method'].str.upper()
    df_eval['Rate'] = df_eval['Rate'].apply(lambda x: f"{x:.2f}")
    
    # Pivot to create the 2D grid
    pivot = df_eval.pivot_table(index='Method', columns='Rate', values='Correct', aggfunc='mean')
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    plt.title(f"Accuracy Heatmap: Method vs. Rate ({model_name})", fontweight='bold')
    plt.ylabel("Poisoning Method")
    plt.xlabel("Poisoning Rate")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def retrain_metalearner(db_path, model_save_path, aim_run, methods_filter=None):
    logger.info(f"--- Retraining Meta-Learner on Augmented DB: {db_path} ---")
    df = pd.read_csv(db_path)
    
    # --- ADDED: Filter by method and dynamically rename the save path ---
    if methods_filter:
        df = df[df['Method'].isin(methods_filter) | (df['Method'] == 'clean')].copy()
        
        # Append methods to the filename (e.g., model_alfa_svm_biggio_svm.joblib)
        base, ext = os.path.splitext(model_save_path)
        model_save_path = f"{base}_{'_'.join(methods_filter)}{ext}"
        logger.info(f"Filtered training to methods: {methods_filter}. Output: {model_save_path}")
    # -------------------------------------------------------------------
    
    drop_cols = ['Data', 'Path', 'Method', 'Rate', 'Is_Poisoned', 'error']
    drop_cols += [c for c in df.columns if c in ['Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison']]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols]
    y = df['Is_Poisoned']
    groups = df['Data']
    
    # Capture Metadata for granular testing before dropping
    methods = df['Method']
    rates = df['Rate']
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Extract test metadata
    y_test_vals = y_test.values
    methods_test = methods.iloc[test_idx].values
    rates_test = rates.iloc[test_idx].values
    
    logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    plots_dir = os.path.join(os.path.dirname(model_save_path), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ==========================================
    # 1. Random Forest
    # ==========================================
    logger.info("Training Random Forest...")
    clf_rf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    y_prob_rf = clf_rf.predict_proba(X_test)[:, 1]
    
    acc_rf, auc_rf, f1_rf = accuracy_score(y_test, y_pred_rf), roc_auc_score(y_test, y_prob_rf), f1_score(y_test, y_pred_rf)
    logger.info(f"[Random Forest] Acc: {acc_rf:.4f} | AUC: {auc_rf:.4f} | F1: {f1_rf:.4f}")
    
    aim_run.track(acc_rf, name="Accuracy", context={"model": "RandomForest", "subset": "global"})
    aim_run.track(auc_rf, name="ROC_AUC", context={"model": "RandomForest", "subset": "global"})
    
    # Generate and Track RF Visualizations
    rf_cm_path = os.path.join(plots_dir, "rf_confusion_matrix.png")
    plot_confusion_matrix(y_test_vals, y_pred_rf, "Random Forest", rf_cm_path)
    aim_run.track(aim.Image(rf_cm_path), name='Confusion_Matrix', context={"model": "RandomForest"})
    
    rf_method_path = os.path.join(plots_dir, "rf_method_accuracy.png")
    plot_method_accuracy(y_test_vals, y_pred_rf, methods_test, "Random Forest", rf_method_path)
    aim_run.track(aim.Image(rf_method_path), name='Accuracy_by_Method', context={"model": "RandomForest"})
    
    rf_heatmap_path = os.path.join(plots_dir, "rf_accuracy_heatmap.png")
    plot_accuracy_heatmap(y_test_vals, y_pred_rf, methods_test, rates_test, "Random Forest", rf_heatmap_path)
    aim_run.track(aim.Image(rf_heatmap_path), name='Accuracy_Heatmap', context={"model": "RandomForest"})
    
    rf_plot_path = os.path.join(plots_dir, "rf_importances.png")
    plot_feature_importances(clf_rf.feature_importances_, feature_cols, "Random Forest", rf_plot_path)
    aim_run.track(aim.Image(rf_plot_path), name='Feature_Importances', context={"model": "RandomForest"})
    
    joblib.dump(clf_rf, model_save_path)

    # ==========================================
    # 2. XGBoost
    # ==========================================
    logger.info("Training XGBoost...")
    scale_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    clf_xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, 
        scale_pos_weight=scale_weight, random_state=42, eval_metric='auc'
    )
    clf_xgb.fit(X_train, y_train)
    y_pred_xgb = clf_xgb.predict(X_test)
    y_prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
    
    acc_xgb, auc_xgb, f1_xgb = accuracy_score(y_test, y_pred_xgb), roc_auc_score(y_test, y_prob_xgb), f1_score(y_test, y_pred_xgb)
    logger.info(f"[XGBoost] Acc: {acc_xgb:.4f} | AUC: {auc_xgb:.4f} | F1: {f1_xgb:.4f}")
    
    aim_run.track(acc_xgb, name="Accuracy", context={"model": "XGBoost", "subset": "global"})
    aim_run.track(auc_xgb, name="ROC_AUC", context={"model": "XGBoost", "subset": "global"})
    
    # Generate and Track XGB Visualizations
    xgb_cm_path = os.path.join(plots_dir, "xgb_confusion_matrix.png")
    plot_confusion_matrix(y_test_vals, y_pred_xgb, "XGBoost", xgb_cm_path)
    aim_run.track(aim.Image(xgb_cm_path), name='Confusion_Matrix', context={"model": "XGBoost"})
    
    xgb_method_path = os.path.join(plots_dir, "xgb_method_accuracy.png")
    plot_method_accuracy(y_test_vals, y_pred_xgb, methods_test, "XGBoost", xgb_method_path)
    aim_run.track(aim.Image(xgb_method_path), name='Accuracy_by_Method', context={"model": "XGBoost"})
    
    xgb_heatmap_path = os.path.join(plots_dir, "xgb_accuracy_heatmap.png")
    plot_accuracy_heatmap(y_test_vals, y_pred_xgb, methods_test, rates_test, "XGBoost", xgb_heatmap_path)
    aim_run.track(aim.Image(xgb_heatmap_path), name='Accuracy_Heatmap', context={"model": "XGBoost"})
    
    xgb_plot_path = os.path.join(plots_dir, "xgb_importances.png")
    plot_feature_importances(clf_xgb.feature_importances_, feature_cols, "XGBoost", xgb_plot_path)
    aim_run.track(aim.Image(xgb_plot_path), name='Feature_Importances', context={"model": "XGBoost"})
    
    # Log numerical accuracy explicitly for each method to Aim Metrics
    for method in np.unique(methods_test):
        mask = (methods_test == method)
        if sum(mask) > 0:
            method_acc_rf = accuracy_score(y_test_vals[mask], y_pred_rf[mask])
            method_acc_xgb = accuracy_score(y_test_vals[mask], y_pred_xgb[mask])
            aim_run.track(method_acc_rf, name="Accuracy_by_Method", context={"model": "RandomForest", "method": method})
            aim_run.track(method_acc_xgb, name="Accuracy_by_Method", context={"model": "XGBoost", "method": method})
    
    # FIX: Save to a different path so it doesn't overwrite the RF model
    xgb_save_path = model_save_path.replace(".joblib", "_xgb.joblib")
    joblib.dump(clf_xgb, xgb_save_path)
    
    logger.info(f"✅ Models saved to: {model_save_path} and {xgb_save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIVA Meta-Learner Training Pipeline")
    parser.add_argument("--db_path", type=str, default="data/universal_meta_database.csv", help="Path to master DB")
    parser.add_argument("--model_path", type=str, default="data/universal_meta_classifier.joblib", help="Path to save models")
    parser.add_argument("--add_datasets", type=int, default=0, help="Number of new datasets to process")
    parser.add_argument("--source", type=str, default="synthetic", choices=["synthetic", "openml"], help="Dataset source")
    parser.add_argument("--base_folder", type=str, default="data", help="Data storage folder")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU cores for PyMFE")
    parser.add_argument("--retrain_only", action="store_true", help="Skip dataset generation and just retrain")
    parser.add_argument("--description", type=str, default="Meta-Learner Training Run", help="Aim run description")
    parser.add_argument("--n_attacks", type=int, default=4, help="Number of attacks per clean dataset")
    parser.add_argument("--methods", nargs='+', type=str, default=None, help="Filter by methods")
    args = parser.parse_args()

    run = aim.Run(experiment="DIVA_MetaLearner_Training")
    run["hparams"] = vars(args)

    try:
        if args.add_datasets > 0 and not args.retrain_only:
            augment_training_db(args.db_path, args.base_folder, args.add_datasets, n_attacks=args.n_attacks, workers=args.workers, source=args.source, methods=args.methods)

        retrain_metalearner(args.db_path, args.model_path, aim_run=run, methods_filter=args.methods)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
    finally:
        run.close()
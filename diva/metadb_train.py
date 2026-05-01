import os
import argparse
import random
import numpy as np
import pandas as pd
import logging
import joblib
import glob
from pathlib import Path
import openml
import aim
import matplotlib.pyplot as plt
import seaborn as sns
import time
import hashlib
from enum import Enum

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample

# --- Modular Pipeline Imports ---
from scripts.cmeasures import compute_cmeasures
from scripts.meta_db import append_to_db
from scripts.utils.plots import *

# --- Import your Specific Poisoners ---
from scripts.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
from scripts.svm_art.svm_art_generate_metadb import ArtSvmPoisoner
from scripts.svm_biggio.svm_biggio_generate_metadb import BiggioSvmPoisoner
from scripts.svm_feature_collision.svm_featurecollision import FeatureCollisionPoisoner
from scripts.witches_brew.witches_brew_generate_metadb import WitchesBrewPoisoner
from scripts.poison_frogs.poison_frogs_generate_metadb import PoisonFrogsPoisoner
from scripts.bullseye_polytope.bullseye_polytope_generate_metadb import BullseyePolytopePoisoner
from scripts.data_generator.image_fetcher import fetch_and_binarize_images
from scripts.data_generator.openml_fetcher import fetch_openml_datasets

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DIVA_Training_Orchestrator")

# ==========================================
# Task Modality Configuration
# ==========================================
class TaskModality(str, Enum):
    TABULAR_BINARY = "tabular_binary"
    IMAGE_BINARY = "image_binary"
    IMAGE_MULTICLASS = "image_multiclass" # Ready for your future expansion

MODALITY_CONFIG = {
    TaskModality.TABULAR_BINARY: {
        "db_path": "data/meta_db_tabular.csv",
        "model_path": "data/meta_classifier_tabular.joblib",
        "valid_sources": ["synthetic", "openml", None],
        "valid_poisoners": ["alfa_svm", "feature_noise_svm", "random_flip_svm", "feature_collision", "biggio_svm"]
    },
    TaskModality.IMAGE_BINARY: {
        "db_path": "data/meta_db_image.csv",
        "model_path": "data/meta_classifier_image.joblib",
        "valid_sources": ["cifar10", "cifar100", "svhn", "mnist", "fashion_mnist", None],
        "valid_poisoners": ["witches_brew", "poison_frogs", "bullseye_polytope"] 
    },
    TaskModality.IMAGE_MULTICLASS: {
        "db_path": "data/meta_db_image_multi.csv",
        "model_path": "data/meta_classifier_image_multi.joblib",
        "valid_sources": ["cifar10", "cifar100", "svhn", None],
        "valid_poisoners": ["witches_brew", "poison_frogs", "bullseye_polytope"]
    }
}

POISONER_MAP = {
    "alfa_svm": AlfaPoisoner,
    "feature_noise_svm": FeatureNoisePoisoner,
    "random_flip_svm": RandomFlipPoisoner,
    "biggio_svm": BiggioSvmPoisoner,
    "feature_collision": FeatureCollisionPoisoner,
    "witches_brew": WitchesBrewPoisoner,
    "poison_frogs": PoisonFrogsPoisoner,
    "bullseye_polytope": BullseyePolytopePoisoner
}

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


def augment_training_db(config, base_folder, n_datasets, n_attacks, workers, source, methods):
    """Orchestrates creating new data and adding it to the Modality-Specific Training DB"""
    db_path = config["db_path"]
    
    allowed_methods = config["valid_poisoners"]
    if methods:
        methods = [m for m in methods if m in allowed_methods]
    else:
        methods = allowed_methods

    if not methods:
        logger.error(f"No valid methods selected for modality. Allowed: {allowed_methods}")
        return

    # 2. Modality Routing
    if config == MODALITY_CONFIG[TaskModality.TABULAR_BINARY]:
        if source == "openml":
            new_clean_files = fetch_openml_datasets(n_datasets, base_folder, db_path=db_path)
        else: # default to synthetic
            new_clean_files = generate_synthetic_data(n_datasets, base_folder)
            
    elif config == MODALITY_CONFIG[TaskModality.IMAGE_BINARY]:
        # If no source provided, use ALL valid image sources!
        sources_to_use = [source] if source else config["valid_sources"]
        
        new_clean_files = fetch_and_binarize_images(
            sources=sources_to_use, 
            n_max=n_datasets, 
            base_folder=base_folder, 
            db_path=db_path
        )

    #advx_range = np.round(np.arange(0.1, 0.41, 0.05), 2)
    advx_range = [0.01, 0.03, 0.05, 0.08, 0.10]
    
    # Create a pool of ALL possible (method, rate) tuples valid for this modality
    all_possible_attacks = [
        (m, r) for m in POISONER_MAP.keys() 
        if m in methods 
        for r in advx_range
    ]
    
    all_generated_metadata = []

    for file in new_clean_files:
        dataname = Path(file).stem
        all_generated_metadata.append({"Data": dataname, "Path": file, "Method": "clean", "Rate": 0.0, "Is_Poisoned": 0})
        
        # Sample exactly n_attacks combinations from the modality pool
        chosen_attacks = random.sample(all_possible_attacks, min(n_attacks, len(all_possible_attacks)))

        logger.info(f"Chosen attacks for {dataname}: {chosen_attacks}")
        
        attack_plan = {}
        for method, rate in chosen_attacks:
            attack_plan.setdefault(method, []).append(rate)
            
        for method, rates in attack_plan.items():
            poisoner = POISONER_MAP[method](base_folder=base_folder)
            generated_meta = poisoner.apply_poisoning(file, sorted(rates))
            if generated_meta:
                all_generated_metadata.extend(generated_meta)

    paths_to_compute = [m["Path"] for m in all_generated_metadata]
    cmeasures_df = compute_cmeasures(paths_to_compute, workers=workers, db_path=db_path)
    append_to_db(db_path, all_generated_metadata, cmeasures_df)

def retrain_metalearner(db_path, model_save_path, aim_run, methods_filter=None):
    logger.info(f"--- Retraining Meta-Learner on Augmented DB: {db_path} ---")
    df = pd.read_csv(db_path)
    
    if methods_filter:
        df = df[df['Method'].isin(methods_filter) | (df['Method'] == 'clean')].copy()
        base, ext = os.path.splitext(model_save_path)
        model_save_path = f"{base}_{'_'.join(methods_filter)}{ext}"
        logger.info(f"Filtered training to methods: {methods_filter}. Output: {model_save_path}")
    
    drop_cols = ['Data', 'Path', 'Method', 'Rate', 'Is_Poisoned', 'error']
    drop_cols += [c for c in df.columns if c in ['Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison']]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols]
    y = df['Is_Poisoned']
    groups = df['Data']
    
    methods = df['Method']
    rates = df['Rate']
    datasets = df['Data']
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    y_test_vals = y_test.values
    methods_test = methods.iloc[test_idx].values
    rates_test = rates.iloc[test_idx].values
    datasets_test = datasets.iloc[test_idx].values
    
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

    rf_curves_path = os.path.join(plots_dir, "rf_confidence_curves.png")
    plot_all_confidence_curves(datasets_test, methods_test, rates_test, y_prob_rf, "Random Forest", rf_curves_path)
    aim_run.track(aim.Image(rf_curves_path), name='Confidence_Curves', context={"model": "RandomForest"})
    
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

    xgb_curves_path = os.path.join(plots_dir, "xgb_confidence_curves.png")
    plot_all_confidence_curves(datasets_test, methods_test, rates_test, y_prob_xgb, "XGBoost", xgb_curves_path)
    aim_run.track(aim.Image(xgb_curves_path), name='Confidence_Curves', context={"model": "XGBoost"})
    
    for method in np.unique(methods_test):
        mask = (methods_test == method)
        if sum(mask) > 0:
            method_acc_rf = accuracy_score(y_test_vals[mask], y_pred_rf[mask])
            method_acc_xgb = accuracy_score(y_test_vals[mask], y_pred_xgb[mask])
            aim_run.track(method_acc_rf, name="Accuracy_by_Method", context={"model": "RandomForest", "method": method})
            aim_run.track(method_acc_xgb, name="Accuracy_by_Method", context={"model": "XGBoost", "method": method})
    
    xgb_save_path = model_save_path.replace(".joblib", "_xgb.joblib")
    joblib.dump(clf_xgb, xgb_save_path)
    
    logger.info(f"✅ Models saved to: {model_save_path} and {xgb_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIVA Meta-Learner Training Pipeline")
    
    # --- Modality and Routing Arguments ---
    parser.add_argument("--modality", type=str, required=True, choices=[e.value for e in TaskModality], help="The core task modality to train/test.")
    parser.add_argument("--source", type=str, default=None, help="Specific dataset source (e.g., openml, cifar10). Must match modality.")
    
    # --- Standard Arguments ---
    parser.add_argument("--add_datasets", type=int, default=0, help="Number of new datasets to process")
    parser.add_argument("--base_folder", type=str, default="data", help="Data storage folder")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU cores for PyMFE")
    parser.add_argument("--retrain_only", action="store_true", help="Skip dataset generation and just retrain")
    parser.add_argument("--description", type=str, default="Meta-Learner Training Run", help="Aim run description")
    parser.add_argument("--n_attacks", type=int, default=4, help="Number of attacks per clean dataset")
    parser.add_argument("--methods", nargs='+', type=str, default=None, help="Filter by methods")
    
    # --- Path Overrides (Optional) ---
    parser.add_argument("--db_path", type=str, default=None, help="Override path to master DB")
    parser.add_argument("--model_path", type=str, default=None, help="Override path to save models")
    args = parser.parse_args()

    # 1. Load Modality Config
    config = MODALITY_CONFIG[TaskModality(args.modality)]
    
    # 2. Resolve Paths
    db_path = args.db_path if args.db_path else config["db_path"]
    model_path = args.model_path if args.model_path else config["model_path"]

    # 3. Safety Checks
    if args.add_datasets > 0 and not args.retrain_only:
        if args.source not in config["valid_sources"]:
            raise ValueError(f"Source '{args.source}' is invalid for modality '{args.modality}'. Valid sources: {config['valid_sources']}")

    run = aim.Run(experiment=f"DIVA_MetaLearner_{args.modality.upper()}")
    run["hparams"] = vars(args)

    try:
        if args.add_datasets > 0 and not args.retrain_only:
            # Note: We now pass the 'config' dictionary to augment_training_db
            augment_training_db(config, args.base_folder, args.add_datasets, args.n_attacks, args.workers, args.source, args.methods)

        retrain_metalearner(db_path, model_path, aim_run=run, methods_filter=args.methods)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
    finally:
        run.close()
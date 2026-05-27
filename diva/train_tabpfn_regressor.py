import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TwoStage_XGB")

# ==========================================
# Configuration
# ==========================================
TARGETS_FILE = "data_2/regression_targets.csv"
METADB_FILE = "data_2/meta_db_regression.csv"
OUTPUT_PLOT = "data_2/twostage_regression_results.png"

# PyMFE Complexity Measure Prefixes
COMPLEXITY_PREFIXES = (
    'f1', 'f1v', 'f2', 'f3', 'f4',  # Feature-based
    'l1', 'l2', 'l3',               # Linearity-based
    'n1', 'n2', 'n3', 'n4',         # Neighborhood-based
    't2', 't3', 't4',               # Network-based
    'c1', 'c2'                      # Dimensionality-based
)

def evaluate_and_log(name, y_true, y_pred):
    """Helper to log and return metrics for a model stage."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info("-" * 30)
    logger.info(f"📊 {name} Metrics")
    logger.info("-" * 30)
    logger.info(f"RMSE : {rmse:.4f}")
    logger.info(f"MAE  : {mae:.4f}")
    logger.info(f"R²   : {r2:.4f}")
    
    return rmse, mae, r2

def main():
    # 1. Load Data
    logger.info(f"Loading targets from {TARGETS_FILE}")
    targets_df = pd.read_csv(TARGETS_FILE)
    
    logger.info(f"Loading meta-database from {METADB_FILE}")
    meta_df = pd.read_csv(METADB_FILE)
    
    # 2. Merge Data on poisoned files
    # We use poisoned file paths as the join key because, at inference, 
    # we only have access to the C-Measures of the poisoned dataset.
    logger.info("Merging datasets on file paths...")
    merged_df = pd.merge(
        targets_df, 
        meta_df, 
        left_on="poisoned_train_file", 
        right_on="Path", 
        how="inner"
    )
    
    if merged_df.empty:
        logger.error("Merge resulted in an empty DataFrame.")
        return
        
    # 3. Define Features and Targets
    exclude_cols = {
        "original_clean_file", "poisoned_train_file", "method", "requested_rate",
        "clean_accuracy", "poisoned_accuracy", "accuracy_drop", 
        "Data", "Path", "Method", "Rate", "Is_Poisoned"
    }
    
    feature_cols = [
        col for col in merged_df.columns 
        if col not in exclude_cols and col.startswith(COMPLEXITY_PREFIXES)
    ]
    
    X = merged_df[feature_cols]
    
    # We now have THREE targets to track
    y_clean = merged_df["clean_accuracy"]
    y_pois = merged_df["poisoned_accuracy"]
    y_drop = merged_df["accuracy_drop"]  # Only used for final evaluation
    
    groups = merged_df["original_clean_file"] 
    
    logger.info(f"Extracted {len(feature_cols)} Complexity features.")

    # 4. Handle Missing Values
    logger.info("Cleaning missing/infinite values in features...")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    valid_feature_cols = X.columns
    
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=valid_feature_cols)

    # 5. Group-wise Train/Test Split
    logger.info("Performing group-wise train-test split (Test Size = 20%)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    train_idx, test_idx = next(gss.split(X_imputed, y_drop, groups))
    
    # Split Features
    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    
    # Split Targets
    y_train_clean, y_test_clean = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
    y_train_pois, y_test_pois = y_pois.iloc[train_idx], y_pois.iloc[test_idx]
    y_test_drop = y_drop.iloc[test_idx] # We only need this for the final test evaluation
    
    assert len(set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx]))) == 0, "Leakage detected!"

    # 6. Initialize Models
    # Using slightly stronger regularization since we are predicting stable baselines
    model_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1
    }
    
    denoiser = XGBRegressor(**model_params)
    estimator = XGBRegressor(**model_params)

    # 7. Train Stage 1: The Denoiser (Predicts Clean Baseline)
    logger.info("Training Stage 1: Denoiser (Predicting Clean Accuracy)...")
    denoiser.fit(X_train, y_train_clean)
    pred_clean = denoiser.predict(X_test)
    
    # 8. Train Stage 2: The Estimator (Predicts Poisoned Accuracy)
    logger.info("Training Stage 2: Estimator (Predicting Poisoned Accuracy)...")
    estimator.fit(X_train, y_train_pois)
    pred_pois = estimator.predict(X_test)

    # 9. Calculate the Estimated Drop
    # The final prediction is simply the difference between the two models
    pred_drop = pred_clean - pred_pois

    # 10. Evaluate All Stages
    evaluate_and_log("Stage 1: Denoiser (Clean Acc)", y_test_clean, pred_clean)
    evaluate_and_log("Stage 2: Estimator (Poisoned Acc)", y_test_pois, pred_pois)
    evaluate_and_log("Final Output: Estimated Drop", y_test_drop, pred_drop)
    logger.info("-" * 30)

    # 11. Plot Results (1x3 Subplots)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    plot_configs = [
        (axes[0], y_test_clean, pred_clean, "Denoiser: Clean Accuracy Baseline", "royalblue"),
        (axes[1], y_test_pois, pred_pois, "Estimator: Poisoned Accuracy", "darkorange"),
        (axes[2], y_test_drop, pred_drop, "Final Output: Estimated Accuracy Drop", "seagreen")
    ]
    
    for ax, true_vals, pred_vals, title, color in plot_configs:
        ax.scatter(true_vals, pred_vals, alpha=0.7, edgecolors='k', c=color)
        max_val = max(true_vals.max(), pred_vals.max())
        min_val = min(true_vals.min(), pred_vals.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Predicted Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    logger.info(f"Saved evaluation plots to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
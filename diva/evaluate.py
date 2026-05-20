import os
import argparse
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_recall_curve, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from pymfe.mfe import MFE
from tabpfn import TabPFNClassifier

# AIM Integration
from aim import Run, Image

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LOAO_Unified_Benchmark")

# ==============================================================================
# Helper Functions (Refactored to Avoid Duplication)
# ==============================================================================
def extract_base_dataset_name(dataname):
    if "_vs_" in dataname:
        part1 = dataname.split("_vs_")[0]
        return part1.rsplit("_", 1)[0]
    return dataname

def prepare_benchmark_data(db_path, run_type, seed, split_target='Is_Poisoned'):
    """
    Standardized data loading, filtering, and train-test splitting 
    (mimicking the original classification benchmark approach).
    """
    df = pd.read_csv(db_path)
    df['BaseGroup'] = df['Data'].apply(extract_base_dataset_name)

    # feature_noise_svm is considered clean
    df.loc[df['Method'].isin(['feature_noise_svm']), 'Is_Poisoned'] = 0
    
    drop_cols = ['Data', 'Path', 'Method', 'Rate', 'Is_Poisoned', 'error', 'BaseGroup']
    drop_cols += [c for c in df.columns if c in ['Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison']]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    if run_type == "complexity_only":
        complexity_bases = MFE.valid_metafeatures(groups=["complexity"])
        feature_cols = [c for c in feature_cols if c.split('.')[0] in complexity_bases]
        logger.info(f"Filtered down to {len(feature_cols)} complexity features.")
    else:
        logger.info(f"Using all {len(feature_cols)} available features.")

    df[feature_cols] = df[feature_cols].fillna(0)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(df[feature_cols], df[split_target], df['BaseGroup']))
    
    train_df_full = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Leave out specific poisoners as defined in classification
    poisoner_to_leave_out = ["diva_attack_xgb", "diva_attack", "diva_attack2"]
    train_df_full = train_df_full[~(train_df_full['Method'].isin(poisoner_to_leave_out))]
    test_df = test_df[~(test_df['Method'].isin(poisoner_to_leave_out))]

    # Identify methods
    poisoners = sorted([m for m in train_df_full['Method'].unique() if m != 'clean'])
    all_methods = sorted(list(train_df_full['Method'].unique()))
    
    logger.info(f"Identified {len(poisoners)} poisoners to test: {poisoners}")
    
    return train_df_full, test_df, feature_cols, poisoners, all_methods

def plot_target_centric(loao_results, loao_global_results, all_methods, poisoners, metric_name, plots_dir, task_name, run_type, aim_run, y_limit=1.1):
    """
    Standardized target-centric grid plotting (used across Classification, Regression, Contrastive).
    Visually separates benign methods (clean, feature_noise_svm) from adversarial poisoners.
    """
    sns.set_theme(style="whitegrid")
    cols = 3
    rows2 = int(np.ceil(len(all_methods) / cols))
    fig2, axes2 = plt.subplots(rows2, cols, figsize=(6 * cols, 5 * rows2), squeeze=False)
    axes2 = axes2.flatten()
    
    benign_methods = ['clean', 'feature_noise_svm']

    for idx, test_target in enumerate(all_methods):
        ax = axes2[idx]
        scores = [loao_results[omitted].get(test_target, 0) for omitted in poisoners]
        
        # Color logic: Green for benign targets, Blue/Red for adversarial targets
        if test_target in benign_methods: 
            colors = ['#2ecc71'] * len(poisoners)
        else:
            colors = ['#e74c3c' if omitted == test_target else '#3498db' for omitted in poisoners]

        bars = ax.bar(poisoners, scores, color=colors, edgecolor='black', linewidth=0.5)

        # Plot global performance overlay on benign plots
        if test_target in benign_methods:
            global_scores = [loao_global_results[omitted] for omitted in poisoners]
            ax.plot(range(len(poisoners)), global_scores, color='darkorange', marker='o', 
                    linestyle='-', linewidth=2, markersize=6, label=f'Global {metric_name}')
            loc = 'lower left' if metric_name == 'Accuracy' else 'upper left'
            ax.legend(loc=loc, fontsize=9)
        
        # Add (Benign) label to title if applicable
        title_suffix = " (Benign)" if test_target in benign_methods else ""
        ax.set_title(f"{metric_name} on '{test_target}'{title_suffix}", fontsize=12, fontweight='bold')
        ax.set_ylim(0, y_limit)
        ax.set_ylabel(f"Detection {metric_name}", fontsize=10)
        ax.set_xlabel("Method Omitted During Training", fontsize=10)
        ax.set_xticks(range(len(poisoners)))
        ax.set_xticklabels(poisoners, rotation=45, ha='right', fontsize=9)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (y_limit * 0.02), f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for idx in range(len(all_methods), len(axes2)): 
        fig2.delaxes(axes2[idx])
        
    fig2.tight_layout()
    target_plot_path = os.path.join(plots_dir, "loao_target_centric_grid.png")
    fig2.savefig(target_plot_path, dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig2), name="target_centric", context={"task": task_name, "run_type": run_type})
    plt.close(fig2)


# ==============================================================================
# 1. Classification Benchmark
# ==============================================================================
def run_classification_benchmark(db_path, aim_run, workers=4, seed=42, run_type="all_features"):
    logger.info(f"--- Starting Classification LOAO Benchmark [{run_type.upper()}] ---")

    plots_dir = f"data/plots_loao_benchmark_{run_type}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Use centralized data preparation
    train_df_full, test_df, feature_cols, poisoners, all_methods = prepare_benchmark_data(
        db_path, run_type, seed, split_target='Is_Poisoned'
    )
    # Re-append 'None' specific to classification setup (this trains on ALL data)
    poisoners.append("None")
    
    # Train/Test Distribution Plot
    train_stats = train_df_full['Method'].value_counts().rename('Train')
    test_stats = test_df['Method'].value_counts().rename('Test')
    dist_df = pd.concat([train_stats, test_stats], axis=1).fillna(0)
    
    sns.set_theme(style="whitegrid")
    fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dist_df.index))
    width = 0.35
    
    bars1 = ax_dist.bar(x - width/2, dist_df['Train'], width, label='Train Subset', color='#3498db', edgecolor='black')
    bars2 = ax_dist.bar(x + width/2, dist_df['Test'], width, label='Test Subset', color='#e74c3c', edgecolor='black')
    
    ax_dist.set_ylabel('Number of Rows', fontsize=12, fontweight='bold')
    ax_dist.set_title(f'Classification Dataset Composition ({run_type.upper()}): Train vs Test', fontsize=14, fontweight='bold')
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(dist_df.index, rotation=45, ha='right', fontsize=11)
    ax_dist.legend(fontsize=11)
    
    fig_dist.tight_layout()
    dist_plot_path = os.path.join(plots_dir, "train_test_distribution.png")
    fig_dist.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig_dist), name="train_test_distribution", context={"task": "classification", "run_type": run_type})
    plt.close(fig_dist)

    loao_results = {}
    loao_global_results = {}
    all_test_preds = []

    for holdout in poisoners:
        logger.info(f"🚀 Training Classification Model: [BLIND TO {holdout.upper()}]")
        train_df = train_df_full[train_df_full['Method'] != holdout]
        train_df = train_df[~((train_df['Is_Poisoned'] == 1) & (train_df['Rate'] <= 0.05))]
        X_train, y_train = train_df[feature_cols], train_df['Is_Poisoned']
        X_test = test_df[feature_cols]
        
        scale_weight = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))
        clf = TabPFNClassifier(n_estimators=200, balance_probabilities=True)
        clf.fit(X_train, y_train)
        
        test_df_copy = test_df.copy()
        #test_df_copy['Prediction'] = clf.predict(X_test)
        optimal_threshold = 0.35
        test_df_copy['Prediction_Prob'] = clf.predict_proba(X_test)[:, 1]
        test_df_copy['Prediction'] = (test_df_copy['Prediction_Prob'] >= optimal_threshold).astype(int)
        test_df_copy['Prediction_Prob'] = clf.predict_proba(X_test)[:, 1]
        test_df_copy['Holdout'] = holdout
        all_test_preds.append(test_df_copy)
        
        global_acc = accuracy_score(test_df_copy['Is_Poisoned'], test_df_copy['Prediction'])
        loao_global_results[holdout] = global_acc
        
        method_accs = {}
        for method in all_methods:
            mask = test_df_copy['Method'] == method
            if mask.sum() > 0:
                method_accs[method] = accuracy_score(test_df_copy.loc[mask, 'Is_Poisoned'], test_df_copy.loc[mask, 'Prediction'])
            else:
                method_accs[method] = 0.0
                
        loao_results[holdout] = method_accs
        logger.info(f"   => Zero-Shot Accuracy on {holdout}: {method_accs.get(holdout, 0):.2%}\n")

    # Threshold Selection Plot
    combined_preds = pd.concat(all_test_preds)
    precisions, recalls, thresholds = precision_recall_curve(
        combined_preds['Is_Poisoned'], combined_preds['Prediction_Prob']
    )
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    fig_thresh, ax_thresh = plt.subplots(figsize=(10, 6))
    ax_thresh.plot(thresholds, precisions[:-1], label='Precision', color='#2ecc71', linewidth=2)
    ax_thresh.plot(thresholds, recalls[:-1], label='Recall', color='#e74c3c', linewidth=2)
    ax_thresh.plot(thresholds, f1_scores[:-1], label='F1 Score', color='#3498db', linewidth=2, linestyle='--')
    ax_thresh.axvline(x=best_threshold, color='black', linestyle=':', label=f'Best F1 Threshold ({best_threshold:.2f})')
    ax_thresh.set_title(f"Performance vs. Decision Threshold ({run_type.upper()})", fontsize=14, fontweight='bold')
    ax_thresh.set_xlabel("Decision Threshold", fontsize=12)
    ax_thresh.set_ylabel("Score", fontsize=12)
    ax_thresh.set_xlim(0, 1)
    ax_thresh.set_ylim(0, 1.05)
    ax_thresh.legend(loc='lower left')
    plt.tight_layout()
    fig_thresh.savefig(os.path.join(plots_dir, "threshold_selection.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig_thresh), name="threshold_selection", context={"task": "classification", "run_type": run_type})
    plt.close(fig_thresh)

    # =========================================================================
    # NEW: Plots for Model Trained on ALL Data (where Holdout == 'None')
    # =========================================================================
    preds_all_train = combined_preds[combined_preds['Holdout'] == 'None'].copy()
    benign_methods = ['clean', 'feature_noise_svm']
    
    if not preds_all_train.empty:
        # --- Plot 1: Accuracy on All Methods (Poisoners + Clean/Noise) ---
        method_accs_all = []
        for method in all_methods:
            mask = preds_all_train['Method'] == method
            if mask.sum() > 0:
                acc = accuracy_score(preds_all_train.loc[mask, 'Is_Poisoned'], preds_all_train.loc[mask, 'Prediction'])
                method_accs_all.append({'Method': method, 'Accuracy': acc})
        
        df_accs_all = pd.DataFrame(method_accs_all)
        
        # Custom palette: Green for benign datasets, Red for poisoners
        custom_palette = {m: '#2ecc71' if m in benign_methods else '#e74c3c' for m in df_accs_all['Method']}
        
        fig_all, ax_all = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_accs_all, x='Method', y='Accuracy', ax=ax_all, palette=custom_palette, hue='Method', legend=False)
        ax_all.set_title(f"Accuracy of Model Trained on ALL Data ({run_type.upper()})\n(Green = Benign, Red = Poisoner)", fontsize=14, fontweight='bold')
        ax_all.set_ylim(0, 1.1)
        ax_all.set_ylabel("Accuracy", fontsize=12)
        ax_all.set_xlabel("Test Method", fontsize=12)
        ax_all.set_xticklabels(ax_all.get_xticklabels(), rotation=45, ha='right')
        
        # Annotate exact accuracy on top of the bars
        for p in ax_all.patches:
            ax_all.annotate(f"{p.get_height():.2f}", 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), 
                            textcoords='offset points', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        fig_all.savefig(os.path.join(plots_dir, "accuracy_all_train_per_method.png"), dpi=300)
        aim_run.track(Image(fig_all), name="accuracy_all_train_per_method", context={"task": "classification", "run_type": run_type})
        plt.close(fig_all)

        # --- Plot 2: Accuracy per Poisoner broken down by Rate ---
        # Filter out ALL benign methods since they don't have a meaningful poisoning rate
        poison_preds = preds_all_train[~preds_all_train['Method'].isin(benign_methods)].copy()
        
        if not poison_preds.empty:
            rate_accs = []
            for (method, rate), group in poison_preds.groupby(['Method', 'Rate']):
                acc = accuracy_score(group['Is_Poisoned'], group['Prediction'])
                rate_accs.append({'Method': method, 'Rate': rate, 'Accuracy': acc})
                
            df_rate_accs = pd.DataFrame(rate_accs)
            
            # Using grouped barplot for clarity (x=Rate, hue=Method)
            fig_rate, ax_rate = plt.subplots(figsize=(14, 6))
            sns.barplot(data=df_rate_accs, x='Rate', y='Accuracy', hue='Method', ax=ax_rate, palette='tab10')
            
            ax_rate.set_title(f"Accuracy by Poisoning Rate (Model Trained on ALL Data) - {run_type.upper()}", fontsize=14, fontweight='bold')
            ax_rate.set_ylim(0, 1.1)
            ax_rate.set_ylabel("Accuracy", fontsize=12)
            ax_rate.set_xlabel("Poisoning Rate", fontsize=12)
            
            # Place legend outside so it doesn't block the bars
            plt.legend(title='Poisoning Method', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            
            fig_rate.savefig(os.path.join(plots_dir, "accuracy_all_train_by_rate.png"), dpi=300)
            aim_run.track(Image(fig_rate), name="accuracy_all_train_by_rate", context={"task": "classification", "run_type": run_type})
            plt.close(fig_rate)
    # =========================================================================

    # Use Standardized Target-Centric Grid Plotting
    plot_target_centric(
        loao_results, loao_global_results, all_methods, poisoners, 
        metric_name='Accuracy', plots_dir=plots_dir, task_name='classification', 
        run_type=run_type, aim_run=aim_run, y_limit=1.1
    )

def run_multiclass_ood_benchmark(db_path, aim_run, workers=4, seed=42, run_type="all_features"):
    """
    Multi-Class + OOD Detection Paradigm.
    Identifies specific attack signatures and flags unseen zero-day attacks using an OOD threshold.
    """
    logger.info(f"--- Starting Multi-Class OOD Benchmark [{run_type.upper()}] ---")

    plots_dir = f"data/plots_loao_multiclass_ood_{run_type}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load Data
    train_df_full, test_df, feature_cols, poisoners, all_methods = prepare_benchmark_data(
        db_path, run_type, seed, split_target='Method'
    )
    
    # Ensure benign datasets are grouped correctly for multi-class
    train_df_full['Method'] = train_df_full['Method'].replace('feature_noise_svm', 'clean')
    test_df['Method'] = test_df['Method'].replace('feature_noise_svm', 'clean')
    
    # Update poisoners list to exclude the newly grouped benign method
    poisoners = sorted([m for m in train_df_full['Method'].unique() if m != 'clean'])

    ood_results = []
    
    for holdout in poisoners:
        logger.info(f"🚀 Training Multi-Class OOD Model: [ZERO-DAY HOLDOUT: {holdout.upper()}]")
        
        # 2. Prepare Training Data (Blind to the holdout attack)
        train_df = train_df_full[train_df_full['Method'] != holdout]
        
        # Encode string labels to integers for XGBoost
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(train_df['Method'])
        X_train = train_df[feature_cols]
        
        X_test = test_df[feature_cols]
        y_test_true = test_df['Method']
        
        # 3. Train Multi-Class Classifier
        # Using XGBoost here as it scales beautifully to multi-class probabilities natively
        clf = XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=6, 
            objective='multi:softprob',
            random_state=seed, 
            n_jobs=workers
        )
        clf.fit(X_train, y_train_encoded)
        
        # 4. Extract Probabilities and Confidence (Maximum Softmax Probability)
        test_probs = clf.predict_proba(X_test)
        max_probs = np.max(test_probs, axis=1) # The "Confidence" score
        raw_preds = le.inverse_transform(np.argmax(test_probs, axis=1))
        
        test_df_copy = test_df.copy()
        test_df_copy['Max_Prob'] = max_probs
        test_df_copy['Raw_Prediction'] = raw_preds
        test_df_copy['Is_Zero_Day'] = (test_df_copy['Method'] == holdout).astype(int)
        
        # 5. Determine OOD Threshold dynamically 
        # (e.g., 5th percentile of the training set's confidence to allow 5% False OOD rate)
        train_probs = clf.predict_proba(X_train)
        train_max_probs = np.max(train_probs, axis=1)
        ood_threshold = np.percentile(train_max_probs, 5) 
        
        # 6. Apply Threshold: If confidence is below threshold, flag as OOD
        test_df_copy['Final_Prediction'] = np.where(
            test_df_copy['Max_Prob'] < ood_threshold, 
            'OOD_ZERO_DAY', 
            test_df_copy['Raw_Prediction']
        )
        
        # 7. Evaluate Performance
        # --- A. Multi-class Accuracy on Known Classes ---
        known_mask = test_df_copy['Is_Zero_Day'] == 0
        known_acc = accuracy_score(
            test_df_copy.loc[known_mask, 'Method'], 
            test_df_copy.loc[known_mask, 'Final_Prediction']
        )
        
        # --- B. Zero-Day Detection Rate (True Positive Rate for OOD) ---
        zero_day_mask = test_df_copy['Is_Zero_Day'] == 1
        zero_day_detection_rate = (test_df_copy.loc[zero_day_mask, 'Final_Prediction'] == 'OOD_ZERO_DAY').mean()
        
        # --- C. OOD AUROC (Threshold-independent ability to separate known vs unknown) ---
        # Note: AUROC expects a higher score for the positive class. Since Max_Prob is LOWER for OOD,
        # we evaluate AUROC on (1 - Max_Prob) as the "OOD-ness" score.
        ood_auc = roc_auc_score(test_df_copy['Is_Zero_Day'], 1 - test_df_copy['Max_Prob'])
        
        logger.info(f"   => Known Class Accuracy: {known_acc:.2%}")
        logger.info(f"   => Zero-Day '{holdout}' Detection Rate: {zero_day_detection_rate:.2%} (Threshold: {ood_threshold:.2f})")
        logger.info(f"   => OOD Separation AUROC: {ood_auc:.4f}\n")
        
        ood_results.append({
            'Holdout': holdout,
            'Known_Accuracy': known_acc,
            'Zero_Day_Detection_Rate': zero_day_detection_rate,
            'OOD_AUROC': ood_auc
        })

        # 8. Plot OOD Confidence Distribution
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.kdeplot(data=test_df_copy[known_mask], x="Max_Prob", fill=True, color="#3498db", label="Known Classes", ax=ax, clip=(0,1))
        sns.kdeplot(data=test_df_copy[zero_day_mask], x="Max_Prob", fill=True, color="#e74c3c", label=f"Zero-Day ({holdout})", ax=ax, clip=(0,1))
        
        ax.axvline(x=ood_threshold, color='black', linestyle=':', linewidth=2, label=f'OOD Threshold ({ood_threshold:.2f})')
        
        ax.set_title(f"OOD Confidence Distribution (Holdout: {holdout})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Maximum Softmax Probability (Confidence)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        fig_path = os.path.join(plots_dir, f"ood_dist_{holdout}.png")
        fig.savefig(fig_path, dpi=300)
        aim_run.track(Image(fig), name=f"ood_dist_{holdout}", context={"task": "multiclass_ood", "run_type": run_type})
        plt.close(fig)

    # 9. Global Summary Plot
    results_df = pd.DataFrame(ood_results)
    
    fig_summary, ax_summary = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df['Holdout']))
    width = 0.4
    
    ax_summary.bar(x - width/2, results_df['Known_Accuracy'], width, label='Known Class Acc', color='#2ecc71', edgecolor='black')
    ax_summary.bar(x + width/2, results_df['Zero_Day_Detection_Rate'], width, label='Zero-Day Detection Rate', color='#e74c3c', edgecolor='black')
    
    ax_summary.plot(x, results_df['OOD_AUROC'], color='#9b59b6', marker='o', linestyle='-', linewidth=2, markersize=8, label='OOD AUROC')
    
    ax_summary.set_title(f"Multi-Class OOD Benchmark Summary ({run_type.upper()})", fontsize=14, fontweight='bold')
    ax_summary.set_ylabel("Score", fontsize=12)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(results_df['Holdout'], rotation=45, ha='right', fontsize=10)
    ax_summary.set_ylim(0, 1.1)
    ax_summary.legend(loc='lower left')
    
    plt.tight_layout()
    fig_summary.savefig(os.path.join(plots_dir, "ood_global_summary.png"), dpi=300)
    aim_run.track(Image(fig_summary), name="ood_global_summary", context={"task": "multiclass_ood", "run_type": run_type})
    plt.close(fig_summary)

# ==============================================================================
# 3. Regression Benchmark
# ==============================================================================
def run_regression_benchmark(db_path, aim_run, workers=4, seed=42, run_type="all_features"):
    logger.info(f"--- Starting LOAO Regression Benchmark [{run_type.upper()}] ---")

    plots_dir = f"data/plots_loao_regression_{run_type}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Use centralized data preparation (note we split on Rate instead for regression)
    train_df_full, test_df, feature_cols, poisoners, all_methods = prepare_benchmark_data(
        db_path, run_type, seed, split_target='Rate'
    )

    loao_results = {}
    loao_feature_importances = {}
    loao_global_results = {}
    all_test_preds = []

    for holdout in poisoners:
        logger.info(f"🚀 Training Regression Model: [BLIND TO {holdout.upper()}]")
        
        train_df = train_df_full[train_df_full['Method'] != holdout]
        X_train, y_train = train_df[feature_cols], train_df['Rate']
        X_test = test_df[feature_cols]
        
        clf = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=seed, eval_metric='mae', n_jobs=workers)
        clf.fit(X_train, y_train)
        loao_feature_importances[holdout] = clf.feature_importances_
        
        test_df_copy = test_df.copy()
        test_df_copy['Prediction'] = np.clip(clf.predict(X_test), a_min=0.0, a_max=None)
        test_df_copy['Holdout'] = holdout
        all_test_preds.append(test_df_copy)
        
        global_mae = mean_absolute_error(test_df_copy['Rate'], test_df_copy['Prediction'])
        loao_global_results[holdout] = global_mae
        
        method_maes = {}
        for method in all_methods:
            mask = test_df_copy['Method'] == method
            if mask.sum() > 0:
                method_maes[method] = mean_absolute_error(test_df_copy.loc[mask, 'Rate'], test_df_copy.loc[mask, 'Prediction'])
            else:
                method_maes[method] = 0.0
                
        loao_results[holdout] = method_maes
        logger.info(f"   => Zero-Shot MAE on {holdout}: {method_maes.get(holdout, 0):.4f}\n")

    max_mae_observed = max([max(maes.values()) for maes in loao_results.values()])
    y_limit = max_mae_observed * 1.2

    # Use Standardized Target-Centric Grid Plotting (Dynamic upper limit and labels handle MAE properly)
    plot_target_centric(
        loao_results, loao_global_results, all_methods, poisoners, 
        metric_name='MAE', plots_dir=plots_dir, task_name='regression', 
        run_type=run_type, aim_run=aim_run, y_limit=y_limit
    )

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified LOAO Evaluator")
    parser.add_argument("metalearner", choices=["classification", "regression", "multiclass"], help="Type of metalearner to train")
    parser.add_argument("--db_path", type=str, default="data_2/meta_db_universal.csv", help="Path to your populated MetaDB")
    parser.add_argument("--workers", type=int, default=4, help="CPU/DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=60, help="Contrastive Training Epochs")
    parser.add_argument("--filtered", type=str, default=None, help="Feature filtering")
    parser.add_argument("--description", type=str, default="Meta-Learner Training Run", help="Aim run description")
    
    args = parser.parse_args()

    aim_run = Run(experiment="LOAO_Unified_Benchmarks")
    aim_run["hparams"] = vars(args)

    try:
        run_type = "complexity_only" if args.filtered == "complexity" else "all_features"
        logger.info(f"========== RUNNING {run_type.upper().replace('_', ' ')} ==========")
        
        if args.metalearner == "classification":
            run_classification_benchmark(args.db_path, aim_run, args.workers, args.seed, run_type)
        elif args.metalearner == "regression":
            run_regression_benchmark(args.db_path, aim_run, args.workers, args.seed, run_type)
        elif args.metalearner == "multiclass":
            run_multiclass_ood_benchmark(args.db_path, aim_run, args.workers, args.seed, run_type)
            
    except Exception as e:
        logger.error(f"Unified LOAO Benchmark failed: {e}", exc_info=True)
    finally:
        aim_run.close()
        logger.info("Aim run logged and closed successfully.")
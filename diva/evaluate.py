import os
import argparse
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_recall_curve, roc_auc_score, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pymfe.mfe import MFE
from tabpfn import TabPFNClassifier, TabPFNRegressor
import joblib
from scipy.special import softmax
from scipy.stats import entropy
import json
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

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
    df.loc[df['Method'] == 'feature_noise_svm', 'Is_Poisoned'] = 0
    df.loc[df['Method'] == 'feature_noise_svm', 'Method'] = 'clean'
    
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
    Standardized single horizontal bar plot (vertical figure) containing:
    - Blue (#3498db): Classifier trained on ALL data, evaluated on specific poisoner
    - Red (#e74c3c): Classifier trained on LOAO (all except this poisoner), evaluated on specific poisoner
    - Green (#2ecc71): Pristine (clean) dataset accuracy for classifier trained on ALL data
    - Orange (darkorange): Global accuracy for classifier trained on ALL data
    """
    sns.set_theme(style="whitegrid")
    
    # 'None' represents the model trained on ALL data in the run_classification_benchmark
    actual_poisoners = [p for p in poisoners if p != "None"]
    
    # Extract "All Data" evaluations safely
    all_data_results = loao_results.get("None", {})
    all_data_global = loao_global_results.get("None", 0.0)
    
    # Prepare metrics matching the order of actual_poisoners
    blue_scores = [all_data_results.get(p, 0.0) for p in actual_poisoners]
    red_scores = [loao_results.get(p, {}).get(p, 0.0) for p in actual_poisoners]
    
    # Extract overall metrics for Pristine and Global
    clean_score = all_data_results.get("clean", 0.0)
    
    # Dynamically scale height based on number of poisoners to maintain a vertical layout
    fig_height = max(8, len(actual_poisoners) * 0.8 + 3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    y = np.arange(len(actual_poisoners))
    height = 0.35
    
    # 1. Plot Paired Poisoner Bars (Blue vs Red) - Using barh for horizontal bars
    bars_blue = ax.barh(y - height/2, blue_scores, height, color='#3498db', edgecolor='black', label='Trained on All Data')
    bars_red = ax.barh(y + height/2, red_scores, height, color='#e74c3c', edgecolor='black', label='Trained on LOAO (Zero-Shot)')
    
    # 2. Add Extra Bars for Pristine (Green) and Global (Orange) at the bottom
    y_extra_1 = len(actual_poisoners)
    y_extra_2 = len(actual_poisoners) + 1
    
    bar_green = ax.barh(y_extra_1, clean_score, height, color='#2ecc71', edgecolor='black', label='Pristine (Trained on All Data)')
    bar_orange = ax.barh(y_extra_2, all_data_global, height, color='darkorange', edgecolor='black', label='Global (Trained on All Data)')
    
    # Set Y-axis ticks & labels
    all_y = list(y) + [y_extra_1, y_extra_2]
    y_labels = actual_poisoners + ['Pristine', 'Global']
    
    ax.set_yticks(all_y)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight='bold')
    
    # Invert Y-axis so it reads top-to-bottom
    ax.invert_yaxis()
    
    # X-axis limits (using the y_limit variable passed into the function)
    ax.set_xlim(0, y_limit)
    ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f"LOAO Zero-Shot vs All-Data {metric_name} Summary ({run_type})", fontsize=14, fontweight='bold')
    
    # Horizontal separator line to visually separate Specific Poisoners vs Global Summaries
    if len(y) > 0:
        ax.axhline(y[-1] + 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        
    # Place legend neatly outside the plot area
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title="Evaluation Context", title_fontsize=11)
    
    # 3. Add numerical value annotations at the end of every bar
    for bars in [bars_blue, bars_red, bar_green, bar_orange]:
        for bar in bars:
            xval = bar.get_width()
            yval = bar.get_y() + bar.get_height() / 2
            ax.text(xval + (y_limit * 0.015), yval, f"{xval:.2f}", 
                    ha='left', va='center', fontsize=9, fontweight='bold')
                    
    fig.tight_layout()
    
    # Save & Track via Aim
    target_plot_path = os.path.join(plots_dir, "loao_target_centric_summary.png")
    fig.savefig(target_plot_path, dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig), name="target_centric_summary", context={"task": task_name, "run_type": run_type})
    plt.close(fig)


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
    poisoners = ["None"]+poisoners
    
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
        X_train, y_train = train_df[feature_cols], train_df['Is_Poisoned']
        X_test = test_df[feature_cols]
        
        scale_weight = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))
        #clf = XGBClassifier()
        clf = TabPFNClassifier(
            n_estimators=32,
            device='auto',
            random_state=seed
        )
        clf.fit(X_train, y_train)

        if holdout == "None":
            model_path = os.path.join(plots_dir, f"loao_tabpfn_model_{run_type}.joblib")
            joblib.dump(clf, model_path)
            logger.info(f"💾 Saved fully trained ALL DATA model to: {model_path}")
        
        test_df_copy = test_df.copy()
        #test_df_copy['Prediction'] = clf.predict(X_test)
        optimal_threshold = 0.76
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
        logger.info(f"   => Global Accuracy on {holdout}: {global_acc:.2%}\n")

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
    combined_preds = pd.concat(all_test_preds)
    
    # Isolate the probability scores of ONLY the genuinely clean datasets
    clean_probs = combined_preds[combined_preds['Is_Poisoned'] == 0]['Prediction_Prob']
    
    # We want 90% of clean data to be classified as 0. 
    # Therefore, we find the 90th percentile of clean probability scores.
    # Any score above this threshold is flagged as poisoned.
    target_clean_accuracy = 0.90
    best_threshold = np.percentile(clean_probs, target_clean_accuracy * 100)
    
    logger.info(f"Targeting {target_clean_accuracy:.0%} Clean Accuracy -> Calculated Threshold: {best_threshold:.4f}")

    # Now apply this new, stricter threshold
    preds_all_train = combined_preds[combined_preds['Holdout'] == 'None'].copy()
    preds_all_train['Prediction'] = (preds_all_train['Prediction_Prob'] >= best_threshold).astype(int)
    
    preds_all_train = combined_preds[combined_preds['Holdout'] == 'None'].copy()
    benign_methods = ['clean']
    
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
        # NEW: Global vs. Clean Accuracy per Threshold Plot (0.05 Steps)
        # =========================================================================
        thresholds_to_test = np.arange(0.0, 1.05, 0.05)
        global_accs = []
        clean_accs_avg = []

        for t in thresholds_to_test:
            # Apply current threshold (t)
            t_preds = (preds_all_train['Prediction_Prob'] >= t).astype(int)
            
            # 1. Global Accuracy
            global_acc = accuracy_score(preds_all_train['Is_Poisoned'], t_preds)
            global_accs.append(global_acc)
            
            # 2. Clean Accuracy (Now naturally includes feature_noise_svm)
            mask_clean = preds_all_train['Method'] == 'clean'
            
            acc_c = accuracy_score(
                preds_all_train.loc[mask_clean, 'Is_Poisoned'], 
                t_preds[mask_clean]
            ) if mask_clean.sum() > 0 else 0
            
            clean_accs_avg.append(acc_c)

        # Generate the Plot
        fig_acc_thresh, ax_acc_thresh = plt.subplots(figsize=(10, 6))
        
        ax_acc_thresh.plot(thresholds_to_test, global_accs, 
                           label='Global Accuracy', color='darkorange', linewidth=2.5, marker='o')
        ax_acc_thresh.plot(thresholds_to_test, clean_accs_avg, 
                           label='Clean Accuracy', color='#2ecc71', linewidth=2.5, marker='s')
        
        ax_acc_thresh.set_title(f"Accuracy vs. Decision Threshold ({run_type.upper()})", fontsize=14, fontweight='bold')
        ax_acc_thresh.set_xlabel("Decision Threshold (Probability required to flag as Poisoned)", fontsize=12)
        ax_acc_thresh.set_ylabel("Accuracy", fontsize=12)
        ax_acc_thresh.set_xticks(np.arange(0.0, 1.05, 0.05))
        ax_acc_thresh.set_xticklabels([f"{x:.2f}" for x in np.arange(0.0, 1.05, 0.05)], rotation=45)
        ax_acc_thresh.set_xlim(0, 1)
        ax_acc_thresh.set_ylim(0, 1.05)
        
        ax_acc_thresh.legend(loc='lower left', fontsize=11)
        ax_acc_thresh.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        acc_thresh_path = os.path.join(plots_dir, "accuracy_vs_threshold_steps.png")
        fig_acc_thresh.savefig(acc_thresh_path, dpi=300)
        aim_run.track(Image(fig_acc_thresh), name="accuracy_vs_threshold_steps", context={"task": "classification", "run_type": run_type})
        plt.close(fig_acc_thresh)
    # =========================================================================

    # Use Standardized Target-Centric Grid Plotting
    plot_target_centric(
        loao_results, loao_global_results, all_methods, poisoners, 
        metric_name='Accuracy', plots_dir=plots_dir, task_name='classification', 
        run_type=run_type, aim_run=aim_run, y_limit=1.1
    )
def run_multiclass_ood_benchmark(db_path, aim_run, workers=4, seed=42, run_type="all_features", binary_model_path=None, binary_threshold=0.5):
    """
    Two-Stage Cascading Pipeline.
    Stage 1: Binary detector filters Clean vs Poisoned.
    Stage 2: Multiclassifier (trained exclusively on poisons) categorizes the attack family.
    """
    logger.info(f"--- Starting Two-Stage Cascading Benchmark [{run_type.upper()}] ---")

    # Hardcode binary path for testing if not provided
    if binary_model_path is None:
        binary_model_path = f"data/plots_loao_benchmark_{run_type}/loao_tabpfn_model_{run_type}.joblib"

    if not os.path.exists(binary_model_path):
        logger.error(f"Binary model not found at {binary_model_path}. Please provide a valid path.")
        return

    logger.info(f"Loading Binary Detector from {binary_model_path}")
    binary_clf = joblib.load(binary_model_path)

    plots_dir = f"data/plots_loao_cascade_{run_type}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load Data
    train_df_full, test_df, feature_cols, poisoners, all_methods = prepare_benchmark_data(
        db_path, run_type, seed, split_target='Method'
    )
    
    # Update poisoners list to exclude the clean method
    poisoners = sorted([m for m in train_df_full['Method'].unique() if m != 'clean'])

    # 2. Define and Apply Family Mapping
    """family_mapping = {
        'art_svm': 'Family_SVM',
        'poissvm_svm': 'Family_SVM',
        
        'learning_to_confuse': 'Family_Stealth',
        'metapoison': 'Family_Stealth',
        'badnets': 'Family_Stealth',
        'witches_brew': 'Family_Stealth',
        
        'alfa_svm': 'Family_Noise',
        'random_flip_svm': 'Family_Noise',
        
        'feature_collision': 'Family_Feature_Collision',
        'poison_frogs': 'Family_Poison_Frogs',
        'clean': 'clean'
    }"""
    family_mapping = {
        'art_svm': 'Family_SVM',
        'poissvm_svm': 'Family_SVM',
        
        'learning_to_confuse': 'Clean_Label',
        'badnets': 'Clean_Label',

        'metapoison': 'Clean_Label',
        'witches_brew': 'Clean_Label',
        'feature_collision': 'Clean_Label',
        'poison_frogs': 'Clean_Label',
        
        'alfa_svm': 'Label_Flip',
        'random_flip_svm': 'Label_Flip',
        
        'clean': 'clean'
    }

    train_df_full['Family'] = train_df_full['Method'].map(family_mapping)
    test_df['Family'] = test_df['Method'].map(family_mapping)

    # =========================================================================
    # 3. Full Dataset Evaluation (Baseline - No Holdouts)
    # =========================================================================
    logger.info(f"🚀 Training Family Classifier: [BASELINE ALL DATA - NO HOLDOUTS]")
    
    # Filter: Train multiclassifier ONLY on poisoned data
    train_df_poison = train_df_full[train_df_full['Family'] != 'clean'].copy()
    
    le_all = LabelEncoder()
    y_train_all = le_all.fit_transform(train_df_poison['Family'])
    
    clf_multi_all = TabPFNClassifier(n_estimators=32, device='auto', random_state=seed)
    clf_multi_all.fit(train_df_poison[feature_cols], y_train_all)
    
    # TEST PIPELINE (All Data)
    X_test_all = test_df[feature_cols]
    
    # Stage 1: Binary Prediction (1 = Poisoned, 0 = Clean)
    # Note: Assuming binary model predicts 'Is_Poisoned' where class 1 is poison.
    bin_probs_all = binary_clf.predict_proba(X_test_all)[:, 1]
    bin_preds_all = (bin_probs_all >= binary_threshold).astype(int)
    
    # Stage 2: Multiclass Prediction
    multi_probs_all = clf_multi_all.predict_proba(X_test_all)
    multi_preds_all = le_all.inverse_transform(np.argmax(multi_probs_all, axis=1))
    
    test_df_all = test_df.copy()
    
    # MERGE: If Binary says Clean (0), assign 'clean'. Else, assign Multiclass Family.
    test_df_all['Pipeline_Prediction'] = np.where(bin_preds_all == 1, multi_preds_all, 'clean')
    
    all_data_accs = []
    logger.info("   => [ALL DATA] Pipeline Accuracies per Family:")
    for family in test_df_all['Family'].unique():
        mask = test_df_all['Family'] == family
        if mask.sum() > 0:
            acc = accuracy_score(test_df_all.loc[mask, 'Family'], test_df_all.loc[mask, 'Pipeline_Prediction'])
            all_data_accs.append({'Family': family, 'Accuracy': acc})
            logger.info(f"      - {family}: {acc:.2%}")
            
    df_all_accs = pd.DataFrame(all_data_accs)
    
    # Plot Accuracy Per Family
    fig_all_acc, ax_all_acc = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_all_accs, x='Family', y='Accuracy', hue='Family', legend=False, ax=ax_all_acc, palette='viridis')
    ax_all_acc.set_title(f"Pipeline Accuracy Per Family (Trained on ALL Data) - {run_type.upper()}", fontsize=14, fontweight='bold')
    ax_all_acc.set_ylim(0, 1.1)
    ax_all_acc.set_ylabel("Accuracy", fontsize=12)
    ax_all_acc.set_xlabel("Family", fontsize=12)
    plt.setp(ax_all_acc.get_xticklabels(), rotation=45, ha='right')
    
    for p in ax_all_acc.patches:
        ax_all_acc.annotate(f"{p.get_height():.2f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
                        
    plt.tight_layout()
    fig_all_acc.savefig(os.path.join(plots_dir, "cascade_accuracy_all_data.png"), dpi=300)
    aim_run.track(Image(fig_all_acc), name="cascade_accuracy_all_data", context={"task": "cascade_classification", "run_type": run_type})
    plt.close(fig_all_acc)

    # Clustered Confusion Matrix (All Data)
    from sklearn.metrics import confusion_matrix
    valid_families_all = sorted([f for f in test_df_all['Family'].unique() if pd.notna(f)])
    cm_all = confusion_matrix(test_df_all['Family'], test_df_all['Pipeline_Prediction'], labels=valid_families_all)
    cm_df_all = pd.DataFrame(cm_all, index=valid_families_all, columns=valid_families_all)
    
    g_all = sns.clustermap(
        cm_df_all, 
        annot=True, 
        fmt='.2f',
        cmap='Blues',
        standard_scale=0, 
        figsize=(10, 10),
        cbar_pos=(0.02, 0.8, 0.05, 0.18), 
        method='ward' 
    )
    g_all.fig.suptitle(f"Cascade Confusion Matrix (ALL DATA) - {run_type.upper()}", fontsize=16, fontweight='bold', y=1.05)
    g_all.ax_heatmap.set_xlabel("Predicted Pipeline Family", fontsize=12, fontweight='bold')
    g_all.ax_heatmap.set_ylabel("True Family", fontsize=12, fontweight='bold')
    plt.setp(g_all.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(g_all.ax_heatmap.get_yticklabels(), rotation=0)
    
    g_all.savefig(os.path.join(plots_dir, "cascade_clustered_cm_all_data.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(g_all.fig), name="cascade_cm_all_data", context={"task": "cascade_classification", "run_type": run_type})
    plt.close(g_all.fig)

    # =========================================================================
    # 4. Zero-Day Generalization Loop
    # =========================================================================
    ood_results = []
    all_test_preds = [] 
    
    for holdout in poisoners:
        logger.info(f"🚀 Training Cascade Multiclassifier: [HOLDOUT METHOD: {holdout.upper()}]")
        
        # Blind the model to the specific holdout attack AND exclude 'clean'
        train_df = train_df_full[(train_df_full['Method'] != holdout) & (train_df_full['Family'] != 'clean')].copy()
        
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(train_df['Family'])
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        
        # Train Multiclassifier
        clf_multi = TabPFNClassifier(n_estimators=32, device='auto', random_state=seed)
        clf_multi.fit(X_train, y_train_encoded)
        
        # Stage 1: Binary Prediction
        bin_probs = binary_clf.predict_proba(X_test)[:, 1]
        bin_preds = (bin_probs >= binary_threshold).astype(int)
        
        # Stage 2: Multiclass Prediction
        multi_probs = clf_multi.predict_proba(X_test)
        multi_preds = le.inverse_transform(np.argmax(multi_probs, axis=1))
        
        test_df_copy = test_df.copy()
        test_df_copy['Is_Zero_Day'] = (test_df_copy['Method'] == holdout).astype(int)
        test_df_copy['Holdout'] = holdout

        # Merge Pipeline Predictions
        test_df_copy['Pipeline_Prediction'] = np.where(bin_preds == 1, multi_preds, 'clean')
        all_test_preds.append(test_df_copy)
        
        # Evaluate Performance (Only for the known/zero-day POISON classes to track generalization)
        known_poison_mask = (test_df_copy['Is_Zero_Day'] == 0) & (test_df_copy['Family'] != 'clean')
        known_acc = accuracy_score(
            test_df_copy.loc[known_poison_mask, 'Family'], 
            test_df_copy.loc[known_poison_mask, 'Pipeline_Prediction']
        ) if known_poison_mask.sum() > 0 else 0
        
        zero_day_mask = test_df_copy['Is_Zero_Day'] == 1
        zero_day_acc = accuracy_score(
            test_df_copy.loc[zero_day_mask, 'Family'], 
            test_df_copy.loc[zero_day_mask, 'Pipeline_Prediction']
        ) if zero_day_mask.sum() > 0 else 0
        
        # Clean Detection accuracy
        clean_mask = test_df_copy['Family'] == 'clean'
        clean_acc = accuracy_score(
            test_df_copy.loc[clean_mask, 'Family'], 
            test_df_copy.loc[clean_mask, 'Pipeline_Prediction']
        ) if clean_mask.sum() > 0 else 0
        
        logger.info(f"   => Clean Retention Accuracy: {clean_acc:.2%}")
        logger.info(f"   => Known Poison Family Accuracy: {known_acc:.2%}")
        logger.info(f"   => Zero-Day '{holdout}' assigned to correct Family: {zero_day_acc:.2%}\n")
        
        ood_results.append({
            'Holdout': holdout,
            'Holdout_Family': family_mapping.get(holdout, 'Unknown'),
            'Known_Poison_Family_Accuracy': known_acc,
            'Zero_Day_Family_Accuracy': zero_day_acc,
            'Clean_Accuracy': clean_acc
        })

    # =========================================================================
    # 5. Global Summary Plot (Averaged per Family)
    # =========================================================================
    results_df = pd.DataFrame(ood_results)
    
    family_summary_df = results_df.groupby('Holdout_Family')[
        ['Known_Poison_Family_Accuracy', 'Zero_Day_Family_Accuracy', 'Clean_Accuracy']
    ].mean().reset_index()
    
    fig_summary, ax_summary = plt.subplots(figsize=(12, 6))
    x = np.arange(len(family_summary_df['Holdout_Family']))
    width = 0.25
    
    ax_summary.bar(x - width, family_summary_df['Known_Poison_Family_Accuracy'], width, label='Known Poison Acc', color='#3498db', edgecolor='black')
    ax_summary.bar(x, family_summary_df['Zero_Day_Family_Accuracy'], width, label='Zero-Day Poison Acc', color='#e74c3c', edgecolor='black')
    ax_summary.bar(x + width, family_summary_df['Clean_Accuracy'], width, label='Clean Detection Retention', color='#2ecc71', edgecolor='black')
    
    ax_summary.set_title(f"Cascade Pipeline Generalization Summary ({run_type.upper()})", fontsize=14, fontweight='bold')
    ax_summary.set_ylabel("Accuracy", fontsize=12)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(family_summary_df['Holdout_Family'], rotation=45, ha='right', fontsize=11)
    ax_summary.set_ylim(0, 1.1)
    ax_summary.legend(loc='lower left')
    
    plt.tight_layout()
    fig_summary.savefig(os.path.join(plots_dir, "cascade_global_summary.png"), dpi=300)
    aim_run.track(Image(fig_summary), name="cascade_global_summary", context={"task": "cascade_classification", "run_type": run_type})
    plt.close(fig_summary)
    
    # =========================================================================
    # 6. Clustered Confusion Matrix (Aggregated across Holdouts)
    # =========================================================================
    combined_preds = pd.concat(all_test_preds)
    
    valid_families = sorted([f for f in combined_preds['Family'].unique() if pd.notna(f)])
    
    cm = confusion_matrix(combined_preds['Family'], combined_preds['Pipeline_Prediction'], labels=valid_families)
    cm_df = pd.DataFrame(cm, index=valid_families, columns=valid_families)
    
    g = sns.clustermap(
        cm_df, 
        annot=True, 
        fmt='.2f',
        cmap='Blues',
        standard_scale=0, 
        figsize=(10, 10),
        cbar_pos=(0.02, 0.8, 0.05, 0.18), 
        method='ward' 
    )
    
    g.fig.suptitle(f"Aggregated Cascade Confusion Matrix (HOLDOUTS) - {run_type.upper()}", fontsize=16, fontweight='bold', y=1.05)
    g.ax_heatmap.set_xlabel("Predicted Pipeline Family", fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel("True Family", fontsize=12, fontweight='bold')
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    
    clustermap_path = os.path.join(plots_dir, "cascade_clustered_cm_holdouts.png")
    g.savefig(clustermap_path, dpi=300, bbox_inches='tight')
    aim_run.track(Image(g.fig), name="cascade_cm_holdouts", context={"task": "cascade_classification", "run_type": run_type})
    plt.close(g.fig)

# ==============================================================================
# 3. Regression Benchmark (Modified for KL Divergence Target)
# ==============================================================================
def run_regression_benchmark(db_path, aim_run, workers=4, seed=42, run_type="all_features"):
    logger.info(f"--- Starting LOAO Regression Benchmark (Target: KL Divergence) [{run_type.upper()}] ---")

    plots_dir = f"data/plots_loao_regression_kl_{run_type}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Use centralized data preparation 
    train_df_full, test_df, feature_cols, poisoners, all_methods = prepare_benchmark_data(
        db_path, run_type, seed, split_target='Method' 
    )

    # -------------------------------------------------------------------------
    # NEW: Compute KL Divergence Targets (Dataset vs. its Theoretical Clean version)
    # -------------------------------------------------------------------------
    def compute_kl_targets(df, scaler=None):
        # 1. Scale features to prevent massive variables from dominating the Softmax
        if scaler is None:
            scaler = StandardScaler()
            scaled_feats = scaler.fit_transform(df[feature_cols])
        else:
            scaled_feats = scaler.transform(df[feature_cols])
            
        scaled_df = pd.DataFrame(scaled_feats, columns=feature_cols, index=df.index)
        scaled_df['BaseGroup'] = df['BaseGroup'].values
        scaled_df['Method'] = df['Method'].values
        
        # 2. Create a reference map of the Clean distributions per BaseGroup
        # Using .mean() guarantees stability if multiple clean sets share a BaseGroup
        scaled_clean_map = scaled_df[scaled_df['Method'] == 'clean'].groupby('BaseGroup')[feature_cols].mean()

        kl_targets = []
        for idx, row in df.iterrows():
            bg = row['BaseGroup']
            if bg in scaled_clean_map.index:
                # Reference Clean Vector (Q)
                q_vec = scaled_clean_map.loc[bg].values.astype(float)
                q = softmax(q_vec)
                
                # Current Dataset Vector (P)
                p_vec = scaled_df.loc[idx, feature_cols].values.astype(float)
                p = softmax(p_vec)
                
                # Calculate KL Divergence: sum(P * log(P / Q))
                kl = entropy(p, q)
                kl_targets.append(kl)
            else:
                # Fallback if no clean baseline exists for this specific BaseGroup
                kl_targets.append(0.0)
        
        df['KL_Divergence'] = kl_targets
        return df, scaler

    logger.info("Computing KL Divergence targets for all datasets...")
    train_df_full, scaler = compute_kl_targets(train_df_full)
    test_df, _ = compute_kl_targets(test_df, scaler)
    # -------------------------------------------------------------------------

    loao_results = {}
    loao_feature_importances = {}
    loao_global_results = {}
    all_test_preds = []

    for holdout in poisoners:
        logger.info(f"🚀 Training Regression Model: [BLIND TO {holdout.upper()}]")
        
        train_df = train_df_full[train_df_full['Method'] != holdout]
        
        # TARGET is now the computed KL Divergence
        X_train, y_train = train_df[feature_cols], train_df['KL_Divergence']
        X_test = test_df[feature_cols]
        
        clf = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=seed, eval_metric='mae', n_jobs=workers)
        clf.fit(X_train, y_train)
        loao_feature_importances[holdout] = clf.feature_importances_
        
        test_df_copy = test_df.copy()
        
        # Predict the KL Divergence (clipped at 0 since divergence can't be negative)
        test_df_copy['Prediction'] = np.clip(clf.predict(X_test), a_min=0.0, a_max=None)
        test_df_copy['Holdout'] = holdout
        all_test_preds.append(test_df_copy)
        
        # Evaluate against the real KL divergence
        global_mae = mean_absolute_error(test_df_copy['KL_Divergence'], test_df_copy['Prediction'])
        loao_global_results[holdout] = global_mae
        
        method_maes = {}
        for method in all_methods:
            mask = test_df_copy['Method'] == method
            if mask.sum() > 0:
                method_maes[method] = mean_absolute_error(test_df_copy.loc[mask, 'KL_Divergence'], test_df_copy.loc[mask, 'Prediction'])
            else:
                method_maes[method] = 0.0
                
        loao_results[holdout] = method_maes
        logger.info(f"   => Zero-Shot MAE on {holdout}: {method_maes.get(holdout, 0):.4f}\n")

    max_mae_observed = max([max(maes.values()) for maes in loao_results.values()])
    y_limit = max_mae_observed * 1.2

    # Use Standardized Target-Centric Grid Plotting
    plot_target_centric(
        loao_results, loao_global_results, all_methods, poisoners, 
        metric_name='MAE', plots_dir=plots_dir, task_name='regression', 
        run_type=run_type, aim_run=aim_run, y_limit=y_limit
    )
# ==============================================================================
# 4. Regression Benchmark (Target: Accuracy Drop on Downstream Classifiers)
# ==============================================================================
def compute_single_accuracy_drop(clean_path, poison_path, seed=42):
    """
    Evaluates the actual attack success by training classifiers on clean vs poisoned data
    and comparing their performance on a held-out clean test set.
    """
    if clean_path == poison_path or pd.isna(clean_path) or pd.isna(poison_path):
        return 0.0

    try:
        df_c = pd.read_csv(clean_path)
        df_p = pd.read_csv(poison_path)

        # Assuming target is the last column or named 'y'
        y_col_c = 'y' if 'y' in df_c.columns else df_c.columns[-1]
        y_col_p = 'y' if 'y' in df_p.columns else df_p.columns[-1]

        X_c, y_c = df_c.drop(columns=[y_col_c]).values, df_c[y_col_c].values
        X_p, y_p = df_p.drop(columns=[y_col_p]).values, df_p[y_col_p].values

        # Ensure shapes align in case of slight generation mismatches
        min_len = min(len(X_c), len(X_p))
        X_c, y_c = X_c[:min_len], y_c[:min_len]
        X_p, y_p = X_p[:min_len], y_p[:min_len]

        # 1. Split Clean into Train/Test
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_c, y_c, test_size=0.2, random_state=seed)
        
        # 2. Split Poisoned (Discard the Test set to mimic realistic deployment)
        Xp_train, _, yp_train, _ = train_test_split(X_p, y_p, test_size=0.2, random_state=seed)

        # Protect against splits that accidentally contain only 1 class
        if len(np.unique(yc_train)) < 2 or len(np.unique(yp_train)) < 2:
            return 0.0

        # 3. Define the downstream classifiers
        models = [
            SVC(kernel='rbf', random_state=seed),
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=seed),
            RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed)
        ]

        drops = []
        for m in models:
            # Baseline: Train on Clean, Evaluate on Clean Test
            m.fit(Xc_train, yc_train)
            acc_clean = accuracy_score(yc_test, m.predict(Xc_test))

            # Under Attack: Train on Poisoned, Evaluate on Clean Test
            m_pois = clone(m)
            m_pois.fit(Xp_train, yp_train)
            acc_pois = accuracy_score(yc_test, m_pois.predict(Xc_test))

            # Record the drop (clipped at 0 in case the poison randomly helped)
            drops.append(max(0.0, acc_clean - acc_pois))

        # Return the average drop across the 3 classifiers
        return float(np.mean(drops))

    except Exception as e:
        # CHANGED TO ERROR so you can see exactly why it is crashing
        logger.error(f"Failed Drop Calc for {clean_path}: {e}")
        return 0.0

def run_regression_acc_drop_benchmark(db_path, aim_run, workers=4, seed=42, run_type="all_features"):
    logger.info(f"--- Starting LOAO Regression Benchmark (Target: Downstream Acc Drop) [{run_type.upper()}] ---")

    plots_dir = f"data/plots_loao_regression_accdrop_{run_type}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Data Preparation
    train_df_full, test_df, feature_cols, poisoners, all_methods = prepare_benchmark_data(
        db_path, run_type, seed, split_target='Method'
    )

    # Filter out Targeted/Clean-Label attacks (we only want to predict global accuracy drops)
    """targeted_poisoners = [
        'feature_collision', 
        'witches_brew', 
        'poison_frogs', 
        'bullseye_polytope', 
        'metapoison', 
        'badnets'  
    ]"""
    targeted_poisoners = [None]

    train_df_full = train_df_full[~train_df_full['Method'].isin(targeted_poisoners)]
    test_df = test_df[~test_df['Method'].isin(targeted_poisoners)]
    
    # Load DF to construct a reference map of clean paths
    full_df = pd.read_csv(db_path)
    full_df = full_df[~full_df['Method'].isin(targeted_poisoners)]
    full_df['BaseGroup'] = full_df['Data'].apply(extract_base_dataset_name)
    clean_paths = full_df[full_df['Method'] == 'clean'].set_index('BaseGroup')['Path'].to_dict()

    # 2. Compute Target Variables (with Caching)
    cache_file = "data/acc_drop_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            acc_drop_cache = json.load(f)
        valid_paths = set(full_df['Path'])
        # Keep scale 0.0 - 1.0
        acc_drop_cache = {k: v for k, v in acc_drop_cache.items() if k in valid_paths}
    else:
        acc_drop_cache = {}

    def get_targets_for_df(df_subset):
        targets, tasks = [], []
        
        for idx, row in df_subset.iterrows():
            p_path = row['Path']
            if p_path in acc_drop_cache:
                continue
            
            c_path = clean_paths.get(row['BaseGroup'])
            
            if row['Method'] == 'clean':
                acc_drop_cache[p_path] = 0.0
            else:
                tasks.append((p_path, c_path))
                
        if tasks:
            logger.info(f"Computing dynamic accuracy drops for {len(tasks)} un-cached datasets...")
            results = Parallel(n_jobs=workers)(
                delayed(compute_single_accuracy_drop)(c, p, seed) for p, c in tasks
            )
            for (p, c), res in zip(tasks, results):
                acc_drop_cache[p] = res
            
            with open(cache_file, 'w') as f:
                json.dump(acc_drop_cache, f)

        for idx, row in df_subset.iterrows():
            targets.append(acc_drop_cache.get(row['Path'], 0.0))
            
        df_subset['Acc_Drop_Target'] = targets
        return df_subset

    logger.info("Aligning dataset targets (Downstream Accuracy Drop)...")
    train_df_full = get_targets_for_df(train_df_full)
    test_df = get_targets_for_df(test_df)

    logger.info("Filtering out failed attacks (Drop < 5%)...")
    # Preserve clean baseline, but drop ineffective poisoners
    train_df_full = train_df_full[(train_df_full['Method'] == 'clean') | (train_df_full['Acc_Drop_Target'] >= 0.05)]
    test_df = test_df[(test_df['Method'] == 'clean') | (test_df['Acc_Drop_Target'] >= 0.05)]

    # 3. Standard LOAO Execution
    loao_results = {}
    loao_global_results = {}
    all_test_preds = []

    for holdout in poisoners:
        if holdout in targeted_poisoners:
            continue
            
        logger.info(f"🚀 Training Drop Predictor: [BLIND TO {holdout.upper()}]")
        
        train_df = train_df_full[train_df_full['Method'] != holdout]
        
        X_train, y_train = train_df[feature_cols], train_df['Acc_Drop_Target']
        X_test = test_df[feature_cols]
        
        clf = TabPFNRegressor()
        clf.fit(X_train, y_train)
        
        test_df_copy = test_df.copy()
        
        # FIXED: Clipped strictly to 1.0 to match the 0.0-1.0 target scale
        test_df_copy['Prediction'] = np.clip(clf.predict(X_test), a_min=0.0, a_max=1.0)
        test_df_copy['Holdout'] = holdout
        all_test_preds.append(test_df_copy)
        
        global_mae = mean_absolute_error(test_df_copy['Acc_Drop_Target'], test_df_copy['Prediction'])
        loao_global_results[holdout] = global_mae
        
        method_maes = {}
        for method in all_methods:
            mask = test_df_copy['Method'] == method
            if mask.sum() > 0:
                method_maes[method] = mean_absolute_error(test_df_copy.loc[mask, 'Acc_Drop_Target'], test_df_copy.loc[mask, 'Prediction'])
            else:
                method_maes[method] = 0.0
                
        loao_results[holdout] = method_maes
        logger.info(f"   => Zero-Shot MAE on {holdout}: {method_maes.get(holdout, 0):.4f}\n")

    # =========================================================================
    # True vs Predicted Comparison Plots
    # =========================================================================
    combined_preds = pd.concat(all_test_preds)
    
    # Plot 1: Scatter Plot
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=combined_preds, x='Acc_Drop_Target', y='Prediction', hue='Method', alpha=0.7, ax=ax_scatter)
    
    max_val = max(combined_preds['Acc_Drop_Target'].max(), combined_preds['Prediction'].max(), 0.1) 
    ax_scatter.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=2, label='Ideal (True = Predicted)')
    ax_scatter.set_title(f"True vs Predicted Accuracy Drop ({run_type.upper()})", fontsize=14, fontweight='bold')
    ax_scatter.set_xlabel("True Accuracy Drop (0 = Failed Attack)", fontsize=12)
    ax_scatter.set_ylabel("Predicted Accuracy Drop", fontsize=12)
    ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_scatter.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig_scatter.savefig(os.path.join(plots_dir, "true_vs_predicted_scatter.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig_scatter), name="true_vs_predicted_scatter", context={"task": "regression_accdrop", "run_type": run_type})
    plt.close(fig_scatter)

    # Plot 2: Bar Chart
    mean_drops = combined_preds.groupby('Method')[['Acc_Drop_Target', 'Prediction']].mean().reset_index()
    mean_drops_melted = mean_drops.melt(id_vars='Method', var_name='Type', value_name='Drop')
    mean_drops_melted['Type'] = mean_drops_melted['Type'].map({'Acc_Drop_Target': 'True Acc Drop', 'Prediction': 'Predicted Acc Drop'})
    
    fig_bar, ax_bar = plt.subplots(figsize=(14, 6))
    sns.barplot(data=mean_drops_melted, x='Method', y='Drop', hue='Type', ax=ax_bar, palette=['#2ecc71', '#3498db'])
    ax_bar.set_title(f"Average True vs Predicted Accuracy Drop by Method ({run_type.upper()})", fontsize=14, fontweight='bold')
    ax_bar.set_ylabel("Accuracy Drop", fontsize=12)
    ax_bar.set_xlabel("Dataset / Method", fontsize=12)
    ax_bar.set_xticklabels(ax_bar.get_xticklabels(), rotation=45, ha='right')
    ax_bar.legend(title='Metric')
    
    plt.tight_layout()
    fig_bar.savefig(os.path.join(plots_dir, "true_vs_predicted_bar.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig_bar), name="true_vs_predicted_bar", context={"task": "regression_accdrop", "run_type": run_type})
    plt.close(fig_bar)

    # =========================================================================
    # Target Centric Grid Plot
    # =========================================================================
    max_mae_observed = max([max(maes.values()) for maes in loao_results.values()])
    y_limit = max_mae_observed * 1.2

    plot_target_centric(
        loao_results, loao_global_results, all_methods, poisoners, 
        metric_name='Drop Prediction MAE', plots_dir=plots_dir, task_name='regression_accdrop', 
        run_type=run_type, aim_run=aim_run, y_limit=y_limit
    )

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified LOAO Evaluator")
    parser.add_argument("metalearner", choices=["classification", "regression", "multiclass", "regression_accdrop"], help="Type of metalearner to train")
    parser.add_argument("--db_path", type=str, default="data/meta_db_universal.csv", help="Path to your populated MetaDB")
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
            run_multiclass_ood_benchmark(args.db_path, aim_run, args.workers, args.seed, run_type, binary_model_path="data/plots_loao_benchmark_all_features/loao_tabpfn_model_all_features.joblib", binary_threshold=0.76)
        
        elif args.metalearner == "regression_accdrop":
            run_regression_acc_drop_benchmark(args.db_path, aim_run, args.workers, args.seed, run_type)
            
    except Exception as e:
        logger.error(f"Unified LOAO Benchmark failed: {e}", exc_info=True)
    finally:
        aim_run.close()
        logger.info("Aim run logged and closed successfully.")
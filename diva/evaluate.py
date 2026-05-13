import os
import argparse
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor
from pymfe.mfe import MFE
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
# PyTorch Contrastive Components (From evaluate_contrastive.py)
# ==============================================================================
class TripletMetaDataset(Dataset):
    """
    Dynamically generates Triplets for Metric Learning GROUPED BY BASE DATASET:
    Anchor: A Clean version of Dataset X
    Positive: Another Clean version of Dataset X (with forced noise augmentation)
    Negative: A Poisoned version of Dataset X
    """
    def __init__(self, X, y, groups, samples_per_epoch=3000, noise_std=0.02): # Slightly increased noise
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
        self.groups = np.array(groups)
        self.noise_std = noise_std
        self.samples_per_epoch = samples_per_epoch

        self.unique_groups = np.unique(self.groups)
        self.group_to_clean = {}
        self.group_to_poison = {}
        
        valid_groups = []
        y_np = np.array(y) 
        
        for g in self.unique_groups:
            idx_g = np.where(self.groups == g)[0]
            
            clean_idx = idx_g[y_np[idx_g] == 0]
            poison_idx = idx_g[y_np[idx_g] == 1]
            
            if len(clean_idx) > 0 and len(poison_idx) > 0:
                self.group_to_clean[g] = clean_idx
                self.group_to_poison[g] = poison_idx
                valid_groups.append(g)
                
        self.valid_groups = valid_groups
        if not self.valid_groups:
            raise ValueError("No valid groups found containing both clean and poisoned data!")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 1. Pick a random valid BaseGroup
        g = np.random.choice(self.valid_groups)
        
        clean_idx = self.group_to_clean[g]
        poison_idx = self.group_to_poison[g]

        # 2. Anchor: Random clean sample from this group
        a_idx = np.random.choice(clean_idx)
        a = self.X[a_idx]

        # 3. Positive: Another clean sample (or same if only 1 exists)
        if len(clean_idx) > 1:
            p_idx = np.random.choice(clean_idx)
            p = self.X[p_idx]
        else:
            p = a

        # 4. Negative: Random poison sample from the SAME group
        n_idx = np.random.choice(poison_idx)
        n = self.X[n_idx]

        return a, p, n

class MetricEmbeddingNet(nn.Module):
    """
    Maps C-Measures into an embedding space using L2 normalization.
    """
    def __init__(self, input_dim, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        emb = self.net(x)
        # L2 Normalize so embeddings live on a hypersphere
        return F.normalize(emb, p=2, dim=1)

# ==============================================================================
# Helper Functions
# ==============================================================================
def extract_base_dataset_name(dataname):
    """
    Extracts the parent dataset name to prevent data leakage.
    Example: 'hf_mnist_0_vs_1' -> 'hf_mnist'
    """
    if "_vs_" in dataname:
        part1 = dataname.split("_vs_")[0]
        return part1.rsplit("_", 1)[0]
    return dataname


# ==============================================================================
# 1. Classification Benchmark (From evaluate.py)
# ==============================================================================
def run_classification_benchmark(db_path, aim_run, workers=4, seed=42, run_type="all_features"):
    logger.info(f"--- Starting Classification LOAO Benchmark [{run_type.upper()}] ---")

    plots_dir = f"data/plots_loao_benchmark_{run_type}"
    os.makedirs(plots_dir, exist_ok=True)
    
    df = pd.read_csv(db_path)
    df['BaseGroup'] = df['Data'].apply(extract_base_dataset_name)
    
    drop_cols = ['Data', 'Path', 'Method', 'Rate', 'Is_Poisoned', 'error', 'BaseGroup']
    drop_cols += [c for c in df.columns if c in ['Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison']]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    if run_type == "complexity_only":
        complexity_bases = MFE.valid_metafeatures(groups=["complexity"])
        feature_cols = [c for c in feature_cols if c.split('.')[0] in complexity_bases]
        logger.info(f"Filtered down to {len(feature_cols)} complexity features.")
    else:
        logger.info(f"Using all {len(feature_cols)} available features.")
    
    # HARD OVERRIDE
    top_features = [
        'n4.sd', 'n4.mean', 'n2.sd', 'n2.mean', 'n1', 'n3.mean',
        'var_importance.mean', 'var_importance.sd', 'linear_discr.sd', 'naive_bayes.sd', 
        'tree_depth.mean', 't2', 't3', 't4', 'density', 'f3.mean', 'naive_bayes.mean', 'linear_discr.mean'
    ]
    feature_cols = [c for c in feature_cols if c in top_features]
    plots_dir+="_filtered"
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"⚠️ HARD OVERRIDE: Restricted to {len(feature_cols)} Top Features.")

    df[feature_cols] = df[feature_cols].fillna(0)
    
    poisoners = sorted([m for m in df['Method'].unique() if m != 'clean'])
    all_methods = sorted(list(df['Method'].unique()))
    logger.info(f"Identified {len(poisoners)} poisoners to test: {poisoners}")

    alfa_df = df[df['Method'] == 'alfa_svm']
    other_df = df[df['Method'] != 'alfa_svm']
    
    if len(alfa_df) > 0:
        alfa_kept = alfa_df.sample(frac=0.5, random_state=42)
        df = pd.concat([other_df, alfa_kept]).reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(df[feature_cols], df['Is_Poisoned'], df['BaseGroup']))
    
    train_df_full = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Train/Test Distribution Plot
    logger.info("Generating Train/Test Distribution Plot...")
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
    
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax_dist.text(bar.get_x() + bar.get_width()/2, yval + (dist_df['Train'].max() * 0.01), 
                             f"{int(yval)}", ha='center', va='bottom', fontsize=9, fontweight='bold')
                
    fig_dist.tight_layout()
    dist_plot_path = os.path.join(plots_dir, "train_test_distribution.png")
    fig_dist.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    
    # Save to aim
    aim_run.track(Image(fig_dist), name="train_test_distribution", context={"task": "classification", "run_type": run_type})
    plt.close(fig_dist)

    loao_results = {}
    loao_feature_importances = {}
    loao_global_results = {}

    for holdout in poisoners:
        logger.info(f"🚀 Training Classification Model: [BLIND TO {holdout.upper()}]")
        train_df = train_df_full[train_df_full['Method'] != holdout]
        X_train, y_train = train_df[feature_cols], train_df['Is_Poisoned']
        X_test = test_df[feature_cols]
        
        scale_weight = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))
        clf = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6, 
            scale_pos_weight=scale_weight, random_state=seed, 
            eval_metric='auc', n_jobs=workers
        )
        clf.fit(X_train, y_train)
        loao_feature_importances[holdout] = clf.feature_importances_
        
        test_df_copy = test_df.copy()
        test_df_copy['Prediction'] = clf.predict(X_test)
        
        global_acc = accuracy_score(test_df_copy['Is_Poisoned'], test_df_copy['Prediction'])
        loao_global_results[holdout] = global_acc
        
        method_accs = {}
        for method in all_methods:
            mask = test_df_copy['Method'] == method
            if mask.sum() > 0:
                acc = accuracy_score(test_df_copy.loc[mask, 'Is_Poisoned'], test_df_copy.loc[mask, 'Prediction'])
                method_accs[method] = acc
            else:
                method_accs[method] = 0.0
                
        loao_results[holdout] = method_accs
        logger.info(f"   => Zero-Shot Accuracy on {holdout}: {method_accs.get(holdout, 0):.2%}\n")

    # Generate Massive Master Plot
    sns.set_theme(style="whitegrid")
    num_plots = len(poisoners)
    cols = 3
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, holdout in enumerate(poisoners):
        ax = axes[idx]
        accs = [loao_results[holdout].get(m, 0) for m in all_methods]
        colors = ['#e74c3c' if m == holdout else '#2ecc71' if m == 'clean' else '#3498db' for m in all_methods]
            
        bars = ax.bar(all_methods, accs, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f"Model Trained WITHOUT '{holdout}'", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Detection Accuracy", fontsize=10)
        ax.set_xticks(range(len(all_methods)))
        ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    for idx in range(num_plots, len(axes)): fig.delaxes(axes[idx])
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "loao_master_grid.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig), name="loao_master_grid", context={"task": "classification", "run_type": run_type})
    plt.close(fig)

    # Target-Centric Plot
    rows2 = int(np.ceil(len(all_methods) / cols))
    fig2, axes2 = plt.subplots(rows2, cols, figsize=(6 * cols, 5 * rows2), squeeze=False)
    axes2 = axes2.flatten()

    for idx, test_target in enumerate(all_methods):
        ax = axes2[idx]
        accs = [loao_results[omitted].get(test_target, 0) for omitted in poisoners]
        colors = ['#e74c3c' if omitted == test_target else '#3498db' for omitted in poisoners]
        if test_target == 'clean': colors = ['#2ecc71'] * len(poisoners)

        bars = ax.bar(poisoners, accs, color=colors, edgecolor='black', linewidth=0.5)

        if test_target == 'clean':
            global_accs = [loao_global_results[omitted] for omitted in poisoners]
            ax.plot(range(len(poisoners)), global_accs, color='darkorange', marker='o', 
                    linestyle='-', linewidth=2, markersize=6, label='Global Model Accuracy')
            ax.legend(loc='lower left', fontsize=9)
        
        ax.set_title(f"Accuracy on '{test_target}'", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Detection Accuracy", fontsize=10)
        ax.set_xlabel("Method Omitted During Training", fontsize=10)
        ax.set_xticks(range(len(poisoners)))
        ax.set_xticklabels(poisoners, rotation=45, ha='right', fontsize=9)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for idx in range(len(all_methods), len(axes2)): fig2.delaxes(axes2[idx])
    fig2.tight_layout()
    fig2.savefig(os.path.join(plots_dir, "loao_target_evolution_grid.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig2), name="target_centric", context={"task": "classification", "run_type": run_type})
    plt.close(fig2)

    # Feature Importances Plot
    fig3, axes3 = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes3 = axes3.flatten()

    for idx, holdout in enumerate(poisoners):
        ax = axes3[idx]
        importances = loao_feature_importances[holdout]
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True).tail(10)
        bars = feat_imp.plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='black', linewidth=0.5)
        ax.set_title(f"Top 10 Features (Without '{holdout}')", fontsize=12, fontweight='bold')
        ax.set_xlabel("Feature Importance Score", fontsize=10)
        ax.set_xlim(0, feat_imp.max() * 1.25)
        for i, v in enumerate(feat_imp):
            ax.text(v + (feat_imp.max() * 0.02), i, f"{v:.3f}", va='center', fontsize=9, fontweight='bold')

    for idx in range(len(poisoners), len(axes3)): fig3.delaxes(axes3[idx])
    fig3.tight_layout()
    fig3.savefig(os.path.join(plots_dir, "loao_feature_importances_grid.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig3), name="loao_feature_importances_grid", context={"task": "classification", "run_type": run_type})
    plt.close(fig3)


# ==============================================================================
# 2. Contrastive Metric Benchmark (From evaluate_contrastive.py)
# ==============================================================================
def run_contrastive_benchmark(db_path, aim_run, workers=4, seed=42, epochs=30, run_type="all_features"):
    logger.info(f"--- Starting Contrastive LOAO Benchmark [{run_type.upper()}] ---")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    plots_dir = f"data/plots_loao_contrastive_{run_type}"
    os.makedirs(plots_dir, exist_ok=True)
    
    df = pd.read_csv(db_path)
    df['BaseGroup'] = df['Data'].apply(extract_base_dataset_name)
    
    drop_cols = ['Data', 'Path', 'Method', 'Rate', 'Is_Poisoned', 'error', 'BaseGroup']
    drop_cols += [c for c in df.columns if c in ['Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison']]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    if run_type == "complexity_only":
        complexity_bases = MFE.valid_metafeatures(groups=["complexity"])
        feature_cols = [c for c in feature_cols if c.split('.')[0] in complexity_bases]
        
    df[feature_cols] = df[feature_cols].fillna(0)
    
    poisoners = sorted([m for m in df['Method'].unique() if m != 'clean'])
    all_methods = sorted(list(df['Method'].unique()))

    alfa_df = df[df['Method'] == 'alfa_svm']
    other_df = df[df['Method'] != 'alfa_svm']
    if len(alfa_df) > 0:
        alfa_kept = alfa_df.sample(frac=0.5, random_state=42)
        df = pd.concat([other_df, alfa_kept]).reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(df[feature_cols], df['Is_Poisoned'], df['BaseGroup']))
    
    train_df_full = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    loao_results = {}
    loao_global_results = {}
    loao_test_embeddings = {}

    for holdout in poisoners:
        logger.info(f"🚀 Training Metric Model: [BLIND TO {holdout.upper()}]")
        
        train_df = train_df_full[train_df_full['Method'] != holdout]
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        y_train = train_df['Is_Poisoned'].values
        groups_train = train_df['BaseGroup'].values
        
        X_test = scaler.transform(test_df[feature_cols].values)
        
        dataset = TripletMetaDataset(X_train, y_train, groups_train, samples_per_epoch=2000)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=workers if device.type == 'cpu' else 0)
        
        model = MetricEmbeddingNet(input_dim=len(feature_cols), embed_dim=32).to(device)
        
        criterion = nn.TripletMarginLoss(margin=0.4, p=2) 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            for a, p, n in loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                optimizer.zero_grad()
                loss = criterion(model(a), model(p), model(n))
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            emb_train = model(X_train_t).cpu().numpy()
            emb_test = model(X_test_t).cpu().numpy()
            
        # FIX: Replaced KNN with a Balanced SVM.
        # KNN fails when one class outnumbers the other 8-to-1 in density.
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=seed)
        clf.fit(emb_train, y_train)
        
        test_df_copy = test_df.copy()
        test_df_copy['Prediction'] = clf.predict(emb_test)
        
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
        loao_test_embeddings[holdout] = (emb_test, test_df_copy['Method'].values)
        logger.info(f"   => Zero-Shot Accuracy on {holdout}: {method_accs.get(holdout, 0):.2%} (Global: {global_acc:.2%})\n")

    # Generate Target-Centric Plot
    sns.set_theme(style="whitegrid")
    
    cols = 3
    # Use len(all_methods) to ensure 'clean' gets its own subplot
    rows2 = int(np.ceil(len(all_methods) / cols))
    fig2, axes2 = plt.subplots(rows2, cols, figsize=(6 * cols, 5 * rows2), squeeze=False)
    axes2 = axes2.flatten()
    
    for idx, test_target in enumerate(all_methods):
        ax = axes2[idx]
        accs = [loao_results[omitted].get(test_target, 0) for omitted in poisoners]
        
        colors = ['#e74c3c' if omitted == test_target else '#3498db' for omitted in poisoners]
        if test_target == 'clean': 
            colors = ['#2ecc71'] * len(poisoners)

        bars = ax.bar(poisoners, accs, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add the Global Model Accuracy curve on top of the 'clean' bar chart
        if test_target == 'clean':
            global_accs = [loao_global_results[omitted] for omitted in poisoners]
            ax.plot(range(len(poisoners)), global_accs, color='darkorange', marker='o', 
                    linestyle='-', linewidth=2, markersize=6, label='Global Model Accuracy')
            ax.legend(loc='lower left', fontsize=9)

        ax.set_title(f"Target Accuracy: '{test_target}'", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Detection Accuracy", fontsize=10)
        ax.set_xlabel("Method Omitted During Training", fontsize=10)
        ax.set_xticks(range(len(poisoners)))
        ax.set_xticklabels(poisoners, rotation=45, ha='right', fontsize=9)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for idx in range(len(all_methods), len(axes2)): 
        fig2.delaxes(axes2[idx])
        
    fig2.tight_layout()
    target_plot_path = os.path.join(plots_dir, "loao_target_centric_grid.png")
    fig2.savefig(target_plot_path, dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig2), name="target_centric", context={"task": "contrastive", "run_type": run_type})
    plt.close(fig2)

    # Generate PCA Plots
    rows = int(np.ceil(len(poisoners) / cols))
    fig_pca, axes_pca = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes_pca = axes_pca.flatten()
    
    for idx, holdout in enumerate(poisoners):
        ax = axes_pca[idx]
        emb_test, test_methods = loao_test_embeddings[holdout]
        
        pca = PCA(n_components=2, random_state=seed)
        emb_2d = pca.fit_transform(emb_test)
        
        clean_mask = test_methods == 'clean'
        holdout_mask = test_methods == holdout
        seen_poison_mask = (~clean_mask) & (~holdout_mask)
        
        ax.scatter(emb_2d[seen_poison_mask, 0], emb_2d[seen_poison_mask, 1], c='#3498db', alpha=0.5, label='Seen Poisons', edgecolors='k', s=40)
        ax.scatter(emb_2d[clean_mask, 0], emb_2d[clean_mask, 1], c='#2ecc71', alpha=0.8, label='Clean Data', edgecolors='k', s=40, marker='s')
        ax.scatter(emb_2d[holdout_mask, 0], emb_2d[holdout_mask, 1], c='#e74c3c', alpha=1.0, label=f'Unseen ({holdout})', edgecolors='k', s=100, marker='*')
        
        ax.set_title(f"Test Embeddings (Blind to '{holdout}')", fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0: ax.legend(loc='best', fontsize=9)
            
    for idx in range(len(poisoners), len(axes_pca)): fig_pca.delaxes(axes_pca[idx])
    fig_pca.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loao_metric_embeddings_pca.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig_pca), name="contrastive_pca_embeddings", context={"task": "contrastive", "run_type": run_type})
    plt.close(fig_pca)

# ==============================================================================
# 3. Regression Benchmark (From evaluate_regression.py)
# ==============================================================================
def run_regression_benchmark(db_path, aim_run, workers=4, seed=42, run_type="all_features"):
    logger.info(f"--- Starting LOAO Regression Benchmark [{run_type.upper()}] ---")

    plots_dir = f"data/plots_loao_regression_{run_type}"
    os.makedirs(plots_dir, exist_ok=True)
    
    df = pd.read_csv(db_path)
    df['BaseGroup'] = df['Data'].apply(extract_base_dataset_name)
    
    drop_cols = ['Data', 'Path', 'Method', 'Rate', 'Is_Poisoned', 'error', 'BaseGroup']
    drop_cols += [c for c in df.columns if c in ['Train.Clean', 'Test.Clean', 'Train.Poison', 'Test.Poison']]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    if run_type == "complexity_only":
        complexity_bases = MFE.valid_metafeatures(groups=["complexity"])
        feature_cols = [c for c in feature_cols if c.split('.')[0] in complexity_bases]
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    poisoners = sorted([m for m in df['Method'].unique() if m != 'clean'])
    all_methods = sorted(list(df['Method'].unique()))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(df[feature_cols], df['Rate'], df['BaseGroup']))
    
    train_df_full = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Train/Test Dist
    train_stats = train_df_full['Method'].value_counts().rename('Train')
    test_stats = test_df['Method'].value_counts().rename('Test')
    dist_df = pd.concat([train_stats, test_stats], axis=1).fillna(0)
    
    sns.set_theme(style="whitegrid")
    fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dist_df.index))
    width = 0.35
    
    bars1 = ax_dist.bar(x - width/2, dist_df['Train'], width, label='Train Subset', color='#3498db', edgecolor='black')
    bars2 = ax_dist.bar(x + width/2, dist_df['Test'], width, label='Test Subset', color='#e74c3c', edgecolor='black')
    
    ax_dist.set_title(f'Regression Dataset Composition ({run_type.upper()})', fontsize=14, fontweight='bold')
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(dist_df.index, rotation=45, ha='right', fontsize=11)
    ax_dist.legend(fontsize=11)
    fig_dist.tight_layout()
    fig_dist.savefig(os.path.join(plots_dir, "train_test_distribution.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig_dist), name="regression_train_test_distribution", context={"task": "regression", "run_type": run_type})
    plt.close(fig_dist)

    loao_results = {}
    loao_feature_importances = {}
    loao_global_results = {}

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

    # Master Plot
    sns.set_theme(style="whitegrid")
    num_plots = len(poisoners)
    cols = 3
    rows = int(np.ceil(num_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, holdout in enumerate(poisoners):
        ax = axes[idx]
        maes = [loao_results[holdout].get(m, 0) for m in all_methods]
        colors = ['#e74c3c' if m == holdout else '#2ecc71' if m == 'clean' else '#3498db' for m in all_methods]
            
        bars = ax.bar(all_methods, maes, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f"Model Trained WITHOUT '{holdout}'", fontsize=12, fontweight='bold')
        ax.set_ylim(0, y_limit)
        ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
        
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (y_limit * 0.02), f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for idx in range(num_plots, len(axes)): fig.delaxes(axes[idx])
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loao_master_grid_mae.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig), name="regression_master_grid_mae", context={"task": "regression", "run_type": run_type})
    plt.close(fig)

    # Target-Centric
    rows2 = int(np.ceil(len(all_methods) / cols))
    fig2, axes2 = plt.subplots(rows2, cols, figsize=(6 * cols, 5 * rows2), squeeze=False)
    axes2 = axes2.flatten()

    for idx, test_target in enumerate(all_methods):
        ax = axes2[idx]
        maes = [loao_results[omitted].get(test_target, 0) for omitted in poisoners]
        colors = ['#e74c3c' if omitted == test_target else '#3498db' for omitted in poisoners]
        if test_target == 'clean': colors = ['#2ecc71'] * len(poisoners)

        bars = ax.bar(poisoners, maes, color=colors, edgecolor='black', linewidth=0.5)
        if test_target == 'clean':
            ax.plot(range(len(poisoners)), [loao_global_results[omitted] for omitted in poisoners], color='darkorange', marker='o', linewidth=2, label='Global MAE')
            ax.legend(loc='upper left', fontsize=9)

        ax.set_title(f"MAE on '{test_target}'", fontsize=12, fontweight='bold')
        ax.set_ylim(0, y_limit)
        ax.set_xticklabels(poisoners, rotation=45, ha='right', fontsize=9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (y_limit * 0.02), f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for idx in range(len(all_methods), len(axes2)): fig2.delaxes(axes2[idx])
    fig2.tight_layout()
    fig2.savefig(os.path.join(plots_dir, "loao_target_evolution_grid_mae.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig2), name="target_centric", context={"task": "regression", "run_type": run_type})
    plt.close(fig2)

    # Feature Importances
    fig3, axes3 = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes3 = axes3.flatten()

    for idx, holdout in enumerate(poisoners):
        ax = axes3[idx]
        feat_imp = pd.Series(loao_feature_importances[holdout], index=feature_cols).sort_values(ascending=True).tail(10)  
        bars = feat_imp.plot(kind='barh', ax=ax, color='#9b59b6', edgecolor='black', linewidth=0.5)
        ax.set_title(f"Top 10 Features (Without '{holdout}')", fontsize=12, fontweight='bold')
        ax.set_xlim(0, feat_imp.max() * 1.25)
        for i, v in enumerate(feat_imp):
            ax.text(v + (feat_imp.max() * 0.02), i, f"{v:.3f}", va='center', fontsize=9, fontweight='bold')

    for idx in range(len(poisoners), len(axes3)): fig3.delaxes(axes3[idx])
    fig3.tight_layout()
    fig3.savefig(os.path.join(plots_dir, "loao_feature_importances_grid_reg.png"), dpi=300, bbox_inches='tight')
    aim_run.track(Image(fig3), name="regression_feature_importances", context={"task": "regression", "run_type": run_type})
    plt.close(fig3)

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified LOAO Evaluator (Classification, Contrastive, Regression)")
    parser.add_argument("metalearner", choices=["classification", "contrastive", "regression"], help="Type of metalearner to train")
    parser.add_argument("--db_path", type=str, default="data/meta_db_universal.csv", help="Path to your populated MetaDB")
    parser.add_argument("--workers", type=int, default=4, help="CPU/DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=60, help="Contrastive Training Epochs")
    parser.add_argument("--filtered", type = str, default = None, help="Feature filtering")
    parser.add_argument("--description", type=str, default="Meta-Learner Training Run", help="Aim run description")
    
    args = parser.parse_args()

    # Initialize aim Run
    aim_run = Run(experiment="LOAO_Unified_Benchmarks")
    aim_run["hparams"] = vars(args)

    try:
        if args.filtered is None:
            logger.info("========== RUNNING ALL FEATURES ==========")
            if args.metalearner == "classification" :
                run_classification_benchmark(args.db_path, aim_run, args.workers, args.seed, run_type="all_features")
            elif args.metalearner == "contrastive" :
                run_contrastive_benchmark(args.db_path, aim_run, args.workers, args.seed, args.epochs, run_type="all_features")
            elif args.metalearner == "regression" :
                run_regression_benchmark(args.db_path, aim_run, args.workers, args.seed, run_type="all_features")
        
        elif args.filtered == "complexity" :
            logger.info("========== RUNNING COMPLEXITY ONLY ==========")
            if args.metalearner == "classification" :
                run_classification_benchmark(args.db_path, aim_run, args.workers, args.seed, run_type="complexity_only")
            elif args.metalearner == "contrastive" :
                run_contrastive_benchmark(args.db_path, aim_run, args.workers, args.seed, args.epochs, run_type="complexity_only")
            elif args.metalearner == "regression" :
                run_regression_benchmark(args.db_path, aim_run, args.workers, args.seed, run_type="complexity_only")
        
    except Exception as e:
        logger.error(f"Unified LOAO Benchmark failed: {e}", exc_info=True)
    finally:
        aim_run.close()
        logger.info("Aim run logged and closed successfully.")
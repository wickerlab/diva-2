import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.lines import Line2D

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

def plot_all_confidence_curves(datasets, methods, rates, probabilities, model_name, save_path):
    """
    Plots a multi-grid figure where each subplot represents an attack method.
    Within each subplot, it draws a transparent confidence curve for every single test dataset,
    colored by its source (Synthetic, OpenML, CIFAR).
    """
    df = pd.DataFrame({
        'Dataset': datasets,
        'Method': methods,
        'Rate': rates,
        'Confidence': probabilities * 100
    })
    
    # Isolate the clean baseline records (which have Method="clean" and Rate=0.0)
    df_clean = df[df['Rate'] == 0.0].copy()
    
    # Find all unique attack methods (excluding the "clean" tag)
    attack_methods = sorted([m for m in df['Method'].unique() if m != 'clean'])
    
    if not attack_methods:
        return
        
    # Calculate grid size (e.g., 3 methods = 2x2 grid)
    cols = int(np.ceil(np.sqrt(len(attack_methods))))
    rows = int(np.ceil(len(attack_methods) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    # Define our dataset color mapping
    color_map = {
        'Synthetic': '#1f77b4', # Blue
        'OpenML': '#2ca02c',    # Green
        'CIFAR': '#9467bd'      # Purple
    }
    
    for idx, method in enumerate(attack_methods):
        ax = axes[idx]
        df_method = df[df['Method'] == method]
        
        # Plot a curve for every single dataset
        for ds, group in df_method.groupby('Dataset'):
            clean_pt = df_clean[df_clean['Dataset'] == ds]
            
            # Stitch the Rate=0.0 clean point to the Rate>0 attacked points
            if not clean_pt.empty:
                curve = pd.concat([clean_pt, group]).sort_values(by='Rate')
            else:
                curve = group.sort_values(by='Rate')
            
            # Determine the dataset type based on its name
            ds_lower = str(ds).lower()
            if 'cifar' in ds_lower:
                ds_type = 'CIFAR'
            elif 'openml' in ds_lower:
                ds_type = 'OpenML'
            else:
                ds_type = 'Synthetic'
                
            # Use alpha=0.2 so hundreds of lines overlapping become a visible density heatmap
            ax.plot(curve['Rate'], curve['Confidence'], marker='o', markersize=2, 
                    alpha=0.2, color=color_map[ds_type])
        
        ax.set_title(f"{method.upper()}", fontweight='bold')
        ax.set_xlabel("Poisoning Rate")
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(-5, 105)
        ax.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        
    # Hide any unused subplots in the grid
    for i in range(len(attack_methods), len(axes)):
        fig.delaxes(axes[i])
        
    # Build a unified custom legend at the bottom of the entire figure
    custom_lines = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, alpha=0.8),
        Line2D([0], [0], color=color_map['Synthetic'], lw=2, alpha=0.7),
        Line2D([0], [0], color=color_map['OpenML'], lw=2, alpha=0.7),
        Line2D([0], [0], color=color_map['CIFAR'], lw=2, alpha=0.7)
    ]
    fig.legend(custom_lines, ['Decision Threshold (50%)', 'Synthetic', 'OpenML', 'CIFAR/Images'], 
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=12)
        
    plt.suptitle(f"Confidence Trajectories by Method ({model_name})", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

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
"""
This script analyzes the mathematical impact of a specific data poisoning attack on the 
geometry of synthetic datasets. It tracks how various Data Complexity Measures (C-Measures) 
evolve as the poisoning rate increases.
"""

import os
import re
import math
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pymfe.mfe import MFE

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("C-Measure_Plotter")

# Default complexity features to extract and plot
DEFAULT_FEATURES = ["f1", "f2", "n1", "n2", "n3", "l1", "l2"]

def load_data(file_path):
    """Safely loads CSV and standardizes target to binary {0, 1}."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = np.where(y == -1, 0, y)  # Ensure standard binary formatting
    return X, y

def compute_c_measures(X, y, measure_names):
    """Computes specific complexity measures using pymfe."""
    try:
        # Instantiate MFE specifically for the requested features
        mfe = MFE(groups=["complexity"])
        mfe.fit(X, y)
        features, values = mfe.extract()
        
        # Return a dictionary mapping feature names to their computed values
        return dict(zip(features, values))
    except Exception as e:
        logger.error(f"Failed to compute measures: {e}")
        return {feat: np.nan for feat in measure_names}

def plot_cmeasure_trajectories(data_dir, num_files, method, c_measures):
    clean_dir = Path(data_dir) / "clean_data"
    poisoned_dir = Path(data_dir) / "poisoned_data"
    
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean data directory not found: {clean_dir}")

    # 1. Use Regex to strictly find the first 'n' clean dataset files
    clean_pattern = re.compile(r"^f\d{4}_i\d{3}_r\d{3}_n\d{4}_sep\d+\.\d+_\d+\.csv$")
    
    all_clean_files = [f for f in os.listdir(clean_dir) if clean_pattern.match(f)]
    all_clean_files.sort()  # Sort alphabetically for deterministic selection
    selected_clean_files = all_clean_files[:num_files]

    if not selected_clean_files:
        logger.error("No clean files matched the regex pattern.")
        return

    logger.info(f"Selected {len(selected_clean_files)} files for analysis.")

    results = []

    # 2. Process each dataset
    for clean_filename in tqdm(selected_clean_files, desc="Processing Datasets", ncols=100):
        base_name = clean_filename.replace('.csv', '')
        clean_file_path = clean_dir / clean_filename
        
        # A. Compute baseline C-Measures (Rate 0.0)
        X_c, y_c = load_data(clean_file_path)
        base_measures = compute_c_measures(X_c, y_c, c_measures)
        
        row_c = {'Dataset': base_name, 'Rate': 0.0}
        row_c.update(base_measures)
        results.append(row_c)

        # B. Find and compute poisoned variants for the specific method
        poison_pattern = re.compile(rf"^{re.escape(base_name)}_{re.escape(method)}_([0-9]+\.[0-9]+)\.csv$")
        poisoned_candidates = list(poisoned_dir.rglob(f"{base_name}_{method}_*.csv"))
        
        for p_file in poisoned_candidates:
            match = poison_pattern.match(p_file.name)
            if match:
                rate = float(match.group(1))
                X_p, y_p = load_data(p_file)
                p_measures = compute_c_measures(X_p, y_p, c_measures)
                
                row_p = {'Dataset': base_name, 'Rate': rate}
                row_p.update(p_measures)
                results.append(row_p)

    # 3. Format DataFrame
    df_results = pd.DataFrame(results)
    
    # Identify the actual features successfully extracted
    extracted_features = [col for col in df_results.columns if col not in ['Dataset', 'Rate']]
    
    if not extracted_features:
        logger.error("No valid C-Measures were extracted. Check your data.")
        return

    # 4. Plotting Subplots
    logger.info("Generating trajectory subplots...")
    sns.set_theme(style="whitegrid")
    
    n_features = len(extracted_features)
    ncols = 2
    nrows = math.ceil(n_features / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True)
    # Ensure axes is always a flattened array even if it's a 1x1 or 1x2 grid
    axes = np.array(axes).flatten()
    
    palette = "tab10" if num_files <= 10 else "husl"

    for i, feature in enumerate(extracted_features):
        ax = axes[i]
        sns.lineplot(
            data=df_results, 
            x="Rate", 
            y=feature, 
            hue="Dataset", 
            ax=ax, 
            marker="o", 
            linewidth=2,
            markersize=5,
            palette=palette
        )
        
        ax.set_title(f"Trajectory of {feature.upper()}", fontweight='bold')
        ax.set_ylabel(f"{feature.upper()} Value")
        ax.set_xticks(np.sort(df_results['Rate'].unique()))
        
        # Remove individual legends to keep plots clean
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Hide any unused subplots if the number of features is odd
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    # Add a single, shared legend at the top of the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Datasets", loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=min(num_files, 5))
    
    # Add a master title
    fig.suptitle(f"C-Measure Trajectories vs. Poisoning Rate (Attack: {method})", fontsize=16, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    
    # Save the output
    output_img = f"trajectory_subplots_{method}_n{num_files}.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved successfully to: {output_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot C-Measure trajectory subplots for multiple datasets.")
    parser.add_argument("-n", "--num_files", type=int, default=5, help="Number of clean datasets to process.")
    parser.add_argument("-m", "--method", type=str, required=True, help="Poisoning method (e.g., alfa_svm, randomlabelflip_svm, art_svm).")
    parser.add_argument("-c", "--c_measures", type=str, nargs='+', default=DEFAULT_FEATURES, help="List of complexity measures to extract (default: f1 f2 n1 n2 n3 l1 l2).")
    parser.add_argument("-d", "--data_dir", type=str, default="data", help="Base directory containing 'clean_data' and 'poisoned_data'.")
    
    args = parser.parse_args()
    
    plot_cmeasure_trajectories(args.data_dir, args.num_files, args.method, args.c_measures)
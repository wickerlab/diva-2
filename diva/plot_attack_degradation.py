import os
import re
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Visualizer")

def load_data(file_path):
    """Safely loads CSV and splits X (features) and y (target)."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def evaluate_degradation(clean_file_path, poisoned_dir="data/poisoned_data"):
    # 1. Setup paths and regex
    clean_path = Path(clean_file_path)
    base_name = clean_path.stem
    poisoned_dir_path = Path(poisoned_dir)

    if not clean_path.exists():
        raise FileNotFoundError(f"Could not find clean file: {clean_file_path}")

    logger.info(f"Analyzing dataset: {base_name}")

    # Regex explanation:
    # ^ matches start of string
    # {base_name}_ matches the exact dataset name followed by an underscore
    # (.+) captures the method name (e.g., 'alfa_svm', 'randomlabelflip_svm')
    # _([0-9]+\.[0-9]+) captures the rate (e.g., '_0.20')
    # \.csv$ matches the extension at the end
    pattern = re.compile(rf"^{re.escape(base_name)}_(.+)_([0-9]+\.[0-9]+)\.csv$")

    # 2. Get the Clean Baseline
    X_clean, y_clean = load_data(clean_path)
    
    # Create the strict, clean test set
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )

    logger.info("Training clean baseline model...")
    clf_baseline = SVC(kernel='linear')
    clf_baseline.fit(X_train_c, y_train_c)
    baseline_acc = accuracy_score(y_test_c, clf_baseline.predict(X_test_c))
    logger.info(f"Baseline Accuracy: {baseline_acc * 100:.2f}%")

    # 3. Find all poisoned variants using recursive glob (rglob)
    all_files = list(poisoned_dir_path.rglob(f"{base_name}_*.csv"))
    
    if not all_files:
        logger.warning("No poisoned files found for this dataset!")
        return

    results = []
    
    # 4. Loop through and evaluate each poisoned file
    logger.info(f"Found {len(all_files)} poisoned variants. Evaluating...")
    for file in tqdm(all_files, ncols=100):
        match = pattern.match(file.name)
        
        if match:
            method = match.group(1)
            rate = float(match.group(2))
            
            try:
                # Load poisoned data
                X_p, y_p = load_data(file)
                
                # Train compromised model
                clf_compromised = SVC(kernel='linear')
                clf_compromised.fit(X_p, y_p)
                
                # Test on the CLEAN test set
                acc = accuracy_score(y_test_c, clf_compromised.predict(X_test_c))
                
                results.append({
                    "Method": method.replace("_svm", "").replace("injection", "").upper(),
                    "Rate": rate,
                    "Accuracy": acc
                })
            except Exception as e:
                logger.error(f"Failed on {file.name}: {e}")

    # 5. Add the baseline (0% rate) for all methods so the plot starts cohesively
    df_results = pd.DataFrame(results)
    methods = df_results['Method'].unique()
    for m in methods:
        # Check if rate 0.0 already exists, if not, inject the baseline
        if not ((df_results['Method'] == m) & (df_results['Rate'] == 0.0)).any():
            df_results = pd.concat([df_results, pd.DataFrame([{"Method": m, "Rate": 0.0, "Accuracy": baseline_acc}])], ignore_index=True)

    # 6. Plotting
    logger.info("Generating plot...")
    plt.figure(figsize=(10, 6))
    
    # Set a clean Seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create the lineplot
    sns.lineplot(
        data=df_results, 
        x="Rate", 
        y="Accuracy", 
        hue="Method", 
        style="Method", 
        markers=True, 
        dashes=False, 
        linewidth=2.5,
        markersize=8
    )
    
    # Add baseline reference line
    plt.axhline(baseline_acc, color='red', linestyle='--', linewidth=2, label='Clean Baseline')
    
    # Formatting
    plt.title(f"Model Degradation vs. Poisoning Rate\nDataset: {base_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Poisoning Rate", fontsize=12)
    plt.ylabel("Accuracy on Clean Test Set", fontsize=12)
    
    # Ensure X-axis shows percentages nicely
    plt.xticks(np.sort(df_results['Rate'].unique()))
    
    plt.legend(title="Attack Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save and show
    output_img = f"degradation_plot_{base_name}.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot attack degradation for a specific dataset.")
    parser.add_argument("-c", "--clean_file", required=True, type=str, help="Path to the specific clean CSV file.")
    parser.add_argument("-p", "--poison_dir", default="data/poisoned_data", type=str, help="Base directory containing poisoned files.")
    
    args = parser.parse_args()
    
    evaluate_degradation(args.clean_file, args.poison_dir)
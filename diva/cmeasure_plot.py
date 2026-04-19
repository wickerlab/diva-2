import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymfe.mfe import MFE
import re
import glob
import os

# ==========================================
# 1. Configuration (Set your paths & labels here)
# ==========================================
# Use a dictionary where KEY = "Legend Label" and VALUE = "File Pattern"
file_patterns = {
    "ART SVM": "/home/gabriel/Polytechnique/3A/internship/diva-2/diva/data/poisoned_data/art_svm/f0150_i015_r029_n5000_s1_1_art_svm_*.csv",
    "Feature Noise": "/home/gabriel/Polytechnique/3A/internship/diva-2/diva/data/poisoned_data/feature_noise_svm/f0150_i015_r029_n5000_s1_1_featurenoiseinjection_svm_*.csv",
    "Random Flip": "/home/gabriel/Polytechnique/3A/internship/diva-2/diva/data/poisoned_data/random_flip_svm/f0150_i015_r029_n5000_s1_1_randomlabelflip_svm_*.csv",
    "alfa_svm": "/home/gabriel/Polytechnique/3A/internship/diva-2/diva/data/poisoned_data/alfa_svm/f0150_i015_r029_n5000_s1_1_alfa_svm_*.csv"
    # Add as many as you need:
    # "Another Method": "path/to/other/*_other_*.csv"
}

# ==========================================
# 2. Data Processing & MFE Extraction
# ==========================================
results = []

for label, pattern in file_patterns.items():
    csv_files = glob.glob(pattern)
    print(f"[{label}] Found {len(csv_files)} files. Starting extraction...")
    
    for file_path in csv_files:
        # Regex update: Looks for an underscore followed by a decimal number and ".csv" at the end
        # This is more generic so it works for multiple different naming conventions
        match = re.search(r'_(\d+\.\d+)\.csv$', file_path)
        if match:
            rate = float(match.group(1))
        else:
            print(f"  -> Could not extract rate from filename: {os.path.basename(file_path)}. Skipping.")
            continue
            
        print(f"  -> Processing rate {rate:.2f}")
        
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Keep ONLY the last 2000 points 
        #df = df.tail(2000)
        
        # Separate features (X) and labels (y)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Extract Complexity Measures
        mfe = MFE(groups=["complexity"])
        mfe.fit(X, y)
        features, values = mfe.extract()
        
        # Store results, including the Label to differentiate curves later
        res_dict = {
            'Label': label,
            'Poisoning_Rate': rate
        }
        res_dict.update(dict(zip(features, values)))
        results.append(res_dict)

# ==========================================
# 3. Formatting and Plotting
# ==========================================
if not results:
    raise ValueError("No results to plot. Check your file_patterns paths.")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Isolate just the complexity measure names (everything except our metadata columns)
measures = [col for col in results_df.columns if col not in ['Label', 'Poisoning_Rate']]
num_measures = len(measures)

# Determine grid size
cols = 4  
rows = (num_measures + cols - 1) // cols 

fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
axes = axes.flatten()

# Get a color palette to ensure distinct colors for different labels
colors = plt.cm.tab10.colors 
unique_labels = results_df['Label'].unique()

# Plot each complexity measure
for i, measure in enumerate(measures):
    
    # Loop through each attack type/label and plot its specific curve
    for j, label in enumerate(unique_labels):
        # Filter data for this specific curve and sort it by Poisoning Rate
        subset = results_df[results_df['Label'] == label].sort_values(by='Poisoning_Rate')
        
        axes[i].plot(
            subset['Poisoning_Rate'], 
            subset[measure], 
            marker='o', 
            linestyle='-', 
            linewidth=2,
            label=label,
            color=colors[j % len(colors)]
        )
        
    axes[i].set_title(f"{measure.upper()}", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Poisoning Rate", fontsize=10)
    axes[i].set_ylabel("Measure Value", fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend to each subplot
    axes[i].legend(fontsize=8)

# Remove any empty subplots at the bottom right
for i in range(num_measures, len(axes)):
    fig.delaxes(axes[i])

plt.suptitle("Complexity Measures (C-Measures) vs. Poisoning Rate\n(Calculated on Last 2000 Points)", fontsize=16, y=1.02, fontweight='bold')
plt.tight_layout()

# Save and show the plot
output_plot_name = './data/figures/c_measures_multi_comparison.png'
plt.savefig(output_plot_name, bbox_inches='tight', dpi=300)
print(f"\nSuccess! Plot saved locally as '{output_plot_name}'")
plt.show()
import argparse
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

# Import your refactored specific poisoner classes
from scripts.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
from scripts.svm_art.svm_art_generate_metadb import ArtSvmPoisoner
from scripts.svm_biggio.svm_biggio_generate_metadb import BiggioSvmPoisoner
import logging


def generate_synthetic_data(n_sets, folder, mode='high_dim'):
    """
    Generates synthetic data optimized to teach the meta-learner 
    how to handle both 'easy' and moderately 'hard' dense datasets.
    """
    # Capped sample sizes to save massive compute time (SVM scales at O(N^3))
    N_SAMPLES_OPTIONS = np.arange(500, 1501, 250) # 500, 750, 1000, 1250, 1500
    N_CLASSES = 2 

    data_path = os.path.join(folder, "clean_data")
    os.makedirs(data_path, exist_ok=True)

    # Feature sizes mimicking SVD reductions
    feature_ranges = list(range(20, 121, 20)) 
    
    grid = [] 
    for f in feature_ranges:
        
        # Bucket 1: Easy
        grid.append({
            "n_samples": N_SAMPLES_OPTIONS, 
            "n_classes": [2], 
            "n_features": [f],
            "n_informative": [int(f * 0.7), int(f * 0.8), int(f * 0.9)], 
            "n_repeated": [0],
            "weights": [[0.5, 0.5], [0.55, 0.45]], # Mostly balanced
            "flip_y": [0.0, 0.01], 
            "class_sep": [1.5, 2.0, 2.5] 
        })

        # Bucket 2: Medium
        grid.append({
            "n_samples": N_SAMPLES_OPTIONS, 
            "n_classes": [2], 
            "n_features": [f],
            "n_informative": [int(f * 0.4), int(f * 0.5), int(f * 0.6)], 
            "n_repeated": [0],
            "weights": [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]], # Introduce imbalance
            "flip_y": [0.03, 0.05, 0.08], 
            "class_sep": [0.8, 1.0, 1.2] 
        })

        # Bucket 3: Hard
        grid.append({
            "n_samples": N_SAMPLES_OPTIONS, 
            "n_classes": [2], 
            "n_features": [f],
            "n_informative": [int(f * 0.2), int(f * 0.3)], 
            "n_repeated": [0],
            "weights": [[0.5, 0.5], [0.7, 0.3], [0.8, 0.2]], # High imbalance
            "flip_y": [0.10, 0.15, 0.20], 
            "class_sep": [0.3, 0.5, 0.7] 
        })

    param_sets = list(ParameterGrid(grid))
    
    replace = len(param_sets) < n_sets
    selected_indices = np.random.choice(len(param_sets), n_sets, replace=replace)

    generated_files = []

    for idx, i in enumerate(selected_indices):
        params = param_sets[i].copy()
        
        remaining_space = params["n_features"] - params["n_informative"] - params["n_repeated"]
        params["n_redundant"] = np.random.randint(0, max(1, remaining_space))
        params["n_clusters_per_class"] = np.random.randint(1, 3) # Simpler cluster shapes
        params["random_state"] = np.random.randint(1000, 99999)

        # Generate the dense data
        X, y = make_classification(**params)
        
        # Scale data to standard normal distribution
        scaler = StandardScaler() 
        X = scaler.fit_transform(X)

        feature_names = [f"x{j}" for j in range(1, X.shape[1] + 1)]
        df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
        
        # Strictly enforce binary 0 and 1 for downstream pipelines
        df["y"] = np.where(y > 0, 1, 0) 

        # Descriptive Filename
        file_name = "f{:04d}_i{:03d}_r{:03d}_n{:04d}_sep{:.1f}".format(
            params["n_features"],
            params["n_informative"],
            params["n_redundant"],
            params["n_samples"],
            params["class_sep"]
        )

        data_list = glob.glob(os.path.join(data_path, f"{file_name}_*.csv"))
        postfix = str(len(data_list) + 1)

        output_path = os.path.join(data_path, f"{file_name}_{postfix}.csv")
        df.to_csv(output_path, index=False)
        
        if (idx + 1) % 5 == 0:
            print(f"Generated {idx + 1}/{n_sets} files...")

        generated_files.append(output_path)

    return generated_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Main")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nSets", default=10, type=int, help="# of random generated synthetic datasets.")
    parser.add_argument("-f", "--folder", default="data", type=str, help="The output folder.")
    parser.add_argument("-s", "--step", type=float, default=0.05, help="Spacing between poisoning rates.")
    parser.add_argument("-m", "--max", type=float, default=0.41, help="End of interval for poisoning rates.")
    args = parser.parse_args()

    base = args.folder
    os.makedirs(base, exist_ok=True)
    advx_range = np.arange(0, args.max, args.step)

    # Generate synthetic datasets once
    logger.info("Generating synthetic datasets...")
    generated_files = generate_synthetic_data(args.nSets, args.folder)

    # Initialize all your poisoners
    #! TODO: Test without poisoning
    #! The idea is C-Measure doesn't change much even though you have poisoned your dataset
    #! So maybe no need to poison and compute the C-Measure for training your metalearner...
    poisoners = [
        #AlfaPoisoner(base_folder=base),
        #FeatureNoisePoisoner(base_folder=base),
        RandomFlipPoisoner(base_folder=base),
        #PoisSVMPoisoner(base_folder=base),
        #BiggioSvmPoisoner(base_folder=base)
    ]

    # Run the standardized pipeline for each method
    for poisoner in poisoners:
        poisoner.run_pipeline(generated_files, advx_range)
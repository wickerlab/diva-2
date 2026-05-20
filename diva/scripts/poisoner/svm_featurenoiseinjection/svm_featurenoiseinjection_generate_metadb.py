import os
import warnings
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging

from ...utils.utils import open_csv, to_csv
from ...base_poisoner import BasePoisoner

warnings.filterwarnings("ignore")

class FeatureNoisePoisoner(BasePoisoner):
    def __init__(self, base_folder, noise_scale=0.05):
        super().__init__(name="feature_noise_svm", base_folder=base_folder)
        # Scale controls the magnitude of the drift relative to each feature's natural variance
        self.noise_scale = noise_scale 

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        path_poison_data_list = []

        # Calculate the natural standard deviation for each feature dynamically
        # Features with 0 variance (constants) will safely receive 0 noise
        feature_stds = np.std(X, axis=0)

        for rate in advx_range:
            path_poison_data = f"{path_output_base}_featurenoiseinjection_svm_{rate:.2f}.csv"
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
            else:
                self.logger.info(f'     Generating {rate * 100:.0f}% poison data via Feature Noise...')
                
                if rate == 0:
                    to_csv(X, y, cols, path_poison_data)
                else:
                    y = np.where(y == -1, 0, y)
                    X_noisy = X.copy()
                    n_noisy = int(len(X) * rate)
                    
                    if n_noisy > 0:
                        noisy_indices = np.random.choice(len(X), size=n_noisy, replace=False)
                        
                        # Generate noise based on the unique distribution of each column
                        noise = np.random.normal(
                            loc=0.0, 
                            scale=feature_stds * self.noise_scale, 
                            size=(n_noisy, X.shape[1])
                        )
                        
                        X_noisy[noisy_indices] += noise
                    
                    to_csv(X_noisy, y, cols, path_poison_data)
            
            path_poison_data_list.append(path_poison_data)

        metadata_list = []
        for p, r in zip(path_poison_data_list, advx_range):
            metadata_list.append({
                "Data": dataname, 
                "Path": p, 
                "Method": self.name, 
                "Rate": r, 
                "Is_Poisoned": 1 if r > 0 else 0
            })
        return metadata_list
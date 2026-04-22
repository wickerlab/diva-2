import os
import warnings
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging

from ..utils.utils import open_csv, to_csv
from ..base_poisoner import BasePoisoner

warnings.filterwarnings("ignore")

class FeatureNoisePoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="feature_noise_svm", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        y = np.where(y == -1, 0, y)
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.complexity_dir, dataname)

        path_poison_data_list = []

        for rate in advx_range:
            path_poison_data = f"{path_output_base}_featurenoiseinjection_svm_{rate:.2f}.csv"
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
            else:
                self.logger.info(f'     Generating {rate * 100:.0f}% poison data via Feature Noise...')
                X_noisy = X.copy()
                n_noisy = int(len(X) * rate)
                
                if n_noisy > 0:
                    noisy_indices = np.random.choice(len(X), size=n_noisy, replace=False)
                    noise = np.random.normal(0, 3.0, size=(n_noisy, X.shape[1]))
                    X_noisy[noisy_indices] += noise
                
                to_csv(X_noisy, y, cols, path_poison_data)
            
            path_poison_data_list.append(path_poison_data)

        data = {
            "Data": np.tile(dataname, reps=len(advx_range)),
            "Path.Poison": path_poison_data_list,
            "Rate": advx_range
        }
        pd.DataFrame(data).to_csv(self.csv_score, mode='a' if os.path.exists(self.csv_score) else 'w', 
                                  header=not os.path.exists(self.csv_score), index=False)
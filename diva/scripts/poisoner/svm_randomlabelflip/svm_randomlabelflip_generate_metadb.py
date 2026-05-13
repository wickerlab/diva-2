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

class RandomFlipPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="random_flip_svm", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        path_poison_data_list = []

        for rate in advx_range:
            path_poison_data = f"{path_output_base}_randomlabelflip_svm_{rate:.2f}.csv"
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
            else:
                self.logger.info(f'     Generating {rate * 100:.0f}% poison data via Random Flip...')
                if rate == 0:
                    to_csv(X, y, cols, path_poison_data)
                    continue
                # Ensure binary {0, 1}
                y = np.where(y == -1, 0, y) 
                y_flip = y.copy()
                n_flip = int(len(y) * rate)
                
                if n_flip > 0:
                    flip_indices = np.random.choice(len(y), size=n_flip, replace=False)
                    # Flip 0 to 1 and 1 to 0
                    y_flip[flip_indices] = 1 - y_flip[flip_indices]
                
                to_csv(X, y_flip, cols, path_poison_data)
            
            path_poison_data_list.append(path_poison_data)

        # Save purely the generation metadata
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
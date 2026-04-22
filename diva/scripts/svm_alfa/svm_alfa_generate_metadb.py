import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import argparse
from pathlib import Path
import logging

from .utils.alfa import alfa
from ..utils.utils import open_csv, to_csv
from ..base_poisoner import BasePoisoner

warnings.filterwarnings('ignore')

class AlfaPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="alfa_svm", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        y = np.where(y == -1, 0, y)
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.complexity_dir, dataname)

        # Train a single, fast linear surrogate to guide ALFA
        clf = SVC(kernel='linear')
        clf.fit(X, y)

        path_poison_data_list = []

        for rate in advx_range:
            path_poison_data = f'{path_output_base}_alfa_svm_{rate:.2f}.csv'
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
            else:
                self.logger.info(f'     Generating {rate * 100:.0f}% poison data via ALFA...')
                if rate == 0:
                    to_csv(X, y, cols, path_poison_data)
                else:
                    # ALFA internally might expect -1/1, transforming temporarily just for the algo
                    y_alfa = np.where(y == 0, -1, 1)
                    y_flip = alfa(X, y_alfa, rate, svc_params=clf.get_params(), max_iter=20)
                    y_flip = np.where(y_flip == -1, 0, 1)
                    to_csv(X, y_flip, cols, path_poison_data)
                    
            path_poison_data_list.append(path_poison_data)

        data = {
            'Data': np.tile(dataname, reps=len(advx_range)),
            'Path.Poison': path_poison_data_list,
            'Rate': advx_range
        }
        pd.DataFrame(data).to_csv(self.csv_score, mode='a' if os.path.exists(self.csv_score) else 'w', 
                                  header=not os.path.exists(self.csv_score), index=False)
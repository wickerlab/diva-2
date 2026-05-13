import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pathlib import Path
import concurrent.futures

from ...utils.utils import open_csv, to_csv
from ...base_poisoner import BasePoisoner
from .utils.poisoning import PoisoningAttackSVM
from .utils.model import ScikitlearnClassifierSVC

warnings.filterwarnings('ignore')

def _generate_biggio_rate(rate, X, y, X_train, y_train, X_val, y_val, cols, path_output_base):
    """
    Standalone worker function to allow ProcessPool parallelization of the Biggio attack.
    """
    path_poison_data = f'{path_output_base}_biggio_svm_{rate:.2f}.csv'
    n_poison = int(len(X_train) * rate) 
    
    if os.path.exists(path_poison_data):
        print(f'     Rate {rate:.2f}: Already generated. Skipping.')
        return path_poison_data
        
    if n_poison == 0:
        # Kept the exact requested logic for 0.0 rate
        to_csv(X, y, cols, path_poison_data)
        return path_poison_data

    print(f'  Generating {n_poison} points to reach {rate * 100:.0f}% via Biggio...')

    # One-hot encoding for ART
    y_val_ohe = np.eye(2)[y_val.astype(int)]
    y_train_ohe = np.eye(2)[y_train.astype(int)]
    clip_bounds = (float(np.min(X_train)), float(np.max(X_train)))

    # 1. ALWAYS attack the pure, clean baseline
    clf_surrogate = SVC(kernel='linear', C=1.0)
    attack_classifier = ScikitlearnClassifierSVC(model=clf_surrogate, clip_values=clip_bounds, nb_classes=2)
    attack_classifier.fit(X_train, y_train_ohe)
    
    attack = PoisoningAttackSVM(
        classifier=attack_classifier, step=0.01, eps=1.0,
        x_train=X_train, y_train=y_train_ohe,
        x_val=X_val, y_val=y_val_ohe, max_iter=20
    )
    
    # 2. Pick random starting points (use deterministic seed for parallel processes)
    seed_val = (hash(path_output_base) + int(rate * 100)) % (2**32)
    np.random.seed(seed_val)
    
    idx = np.random.choice(len(X_train), min(n_poison, len(X_train)), replace=True)
    initial_poison_x = np.copy(X_train[idx])
    initial_poison_y = 1 - np.copy(y_train[idx]) 
    initial_poison_y_ohe = np.eye(2)[initial_poison_y.astype(int)]
    
    # 3. Generate ALL points at once
    malicious_x, malicious_y_ohe = attack.poison(initial_poison_x, initial_poison_y_ohe)
    malicious_y = np.argmax(malicious_y_ohe, axis=1)
    
    # 4. Save using X_train as strictly requested
    X_final = np.vstack([X_train, malicious_x])
    y_final = np.concatenate([y_train, malicious_y])
    
    to_csv(X_final, y_final, cols, path_poison_data)
    
    return path_poison_data

class BiggioSvmPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="biggio_svm", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        
        MAX_SAMPLES = 500
        if len(X) > MAX_SAMPLES:
            self.logger.warning(f"Dataset has {len(X)} points. Subsampling down to {MAX_SAMPLES} for {self.name} attack.")
            np.random.seed(42)  # Deterministic downsample
            sub_indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
            X = X[sub_indices]
            y = y[sub_indices]
            
        y = np.where(y == -1, 0, y)
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        self.logger.warning("Using specific 20/80 train/val split for surrogate gradients.")
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=42)

        self.logger.info(f"Dispatching Biggio rates {advx_range} in parallel...")
        
        # Dispatch the attack calculations concurrently
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    _generate_biggio_rate, 
                    rate, X, y, X_train, y_train, X_val, y_val, cols, path_output_base
                ): rate for rate in advx_range
            }
            
            # Wait for all processes to finish and catch any potential errors
            for future in concurrent.futures.as_completed(futures):
                rate = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(f'Biggio attack failed for rate {rate} on {dataname}: {exc}')

        # Reconstruct the deterministic path list for the MetaDB
        path_poison_data_list = [f'{path_output_base}_biggio_svm_{rate:.2f}.csv' for rate in advx_range]

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
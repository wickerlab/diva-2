"""
Clean-Label Feature Collision Attack (Poison Frogs!)
Citation: Shafahi, A., et al. (2018). "Poison Frogs! Targeted Clean-Label 
Poisoning Attacks on Neural Networks." NeurIPS.

Adapted for tabular/feature-space data. Generates clean-label poisons by 
creating a convex combination (collision) between base instances and target 
instances in the feature space.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import concurrent.futures

from ..utils.utils import open_csv, to_csv
from ..base_poisoner import BasePoisoner

warnings.filterwarnings('ignore')

def _generate_feature_collision_rate(rate, X, y, cols, path_output_base, alpha=0.7):
    """
    Standalone worker function to allow parallelization of the Feature Collision attack.
    
    Args:
        alpha (float): The blending coefficient. 
                       If 1.0, poison equals the target exactly (pure collision).
                       If 0.7, poison is 70% target, 30% base (highly effective, stealthier).
    """
    path_poison_data = f'{path_output_base}_feature_collision_{rate:.2f}.csv'
    n_poison = int(len(X) * rate) 
    
    if os.path.exists(path_poison_data):
        print(f'     Rate {rate:.2f}: Already generated. Skipping.')
        return path_poison_data
        
    if n_poison == 0:
        to_csv(X, y, cols, path_poison_data)
        return path_poison_data

    # Deterministic seed for reproducible parallel generation
    seed_val = (hash(path_output_base) + int(rate * 100)) % (2**32)
    np.random.seed(seed_val)

    # Separate classes
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    if len(idx_0) == 0 or len(idx_1) == 0:
        raise ValueError("Dataset must contain both classes to perform feature collision.")

    # We will split the attack symmetrically: 
    # Half the poisons pull Class 1 towards Class 0, half pull Class 0 towards Class 1.
    n_pois_0 = n_poison // 2
    n_pois_1 = n_poison - n_pois_0

    # --- 1. Attack Class 1 (Target=1, Base=0, Poison Label=0) ---
    targ_idx_1 = np.random.choice(idx_1, n_pois_0, replace=True)
    base_idx_0 = np.random.choice(idx_0, n_pois_0, replace=True)
    
    targets_1 = X[targ_idx_1]
    bases_0 = X[base_idx_0]
    
    # Analytical Feature Collision: p = alpha * target + (1 - alpha) * base
    pois_x_0 = (alpha * targets_1) + ((1 - alpha) * bases_0)
    pois_y_0 = np.zeros(n_pois_0) # Clean label of the base

    # --- 2. Attack Class 0 (Target=0, Base=1, Poison Label=1) ---
    targ_idx_0 = np.random.choice(idx_0, n_pois_1, replace=True)
    base_idx_1 = np.random.choice(idx_1, n_pois_1, replace=True)
    
    targets_0 = X[targ_idx_0]
    bases_1 = X[base_idx_1]
    
    pois_x_1 = (alpha * targets_0) + ((1 - alpha) * bases_1)
    pois_y_1 = np.ones(n_pois_1) # Clean label of the base

    # Combine new malicious points
    malicious_x = np.vstack([pois_x_0, pois_x_1])
    malicious_y = np.concatenate([pois_y_0, pois_y_1])

    # Append to the clean dataset
    X_final = np.vstack([X, malicious_x])
    y_final = np.concatenate([y, malicious_y])

    to_csv(X_final, y_final, cols, path_poison_data)
    
    return path_poison_data


class FeatureCollisionPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="feature_collision", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        y = np.where(y == -1, 0, y) # Ensure strict 0/1 binarization
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        self.logger.info(f"Dispatching Feature Collision rates {advx_range} in parallel...")
        
        # Dispatch the attack calculations concurrently
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    _generate_feature_collision_rate, 
                    rate, X, y, cols, path_output_base, alpha=0.75
                ): rate for rate in advx_range
            }
            
            for future in concurrent.futures.as_completed(futures):
                rate = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self.logger.error(f'Feature Collision attack failed for rate {rate} on {dataname}: {exc}')

        path_poison_data_list = [f'{path_output_base}_feature_collision_{rate:.2f}.csv' for rate in advx_range]

        # Strictly return the list of dictionaries expected by the MetaDB handler
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
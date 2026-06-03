import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from tqdm import tqdm
import concurrent.futures

from ...base_poisoner import BasePoisoner
from ...utils.utils import open_csv, to_csv

RANDOM_SEED = 100
MAX_ITERATIONS = 25
EPSILON = 1e-6

def _generate_poissvm_rate(rate, X_full, y_full, cols, path_output_base):
    path_poison_data = f"{path_output_base}_poissvm_svm_{rate:.2f}.csv"
    n_poison = int(len(X_full) * rate)

    if os.path.exists(path_poison_data):
        return path_poison_data
        
    if n_poison == 0:
        to_csv(X_full, y_full, cols, path_poison_data)
        return path_poison_data

    # --- 1. REPLACEMENT STRATEGY ---
    # Ensure final dataset size exactly matches clean dataset size
    seed_val = (hash(path_output_base) + int(rate * 100)) % (2**32)
    np.random.seed(seed_val)
    
    idx_all = np.arange(len(X_full))
    idx_poison = np.random.choice(idx_all, n_poison, replace=False)
    idx_clean = np.setdiff1d(idx_all, idx_poison)
    
    X_clean_remainder = X_full[idx_clean]
    y_clean_remainder = y_full[idx_clean]
    
    # --- 2. MICRO-SURROGATE BOUNDING ---
    # We cap the background to ensure the 25x retraining loop doesn't stall
    MAX_COMPUTE_LIMIT = 200
    surrogate_size = min(len(X_clean_remainder), MAX_COMPUTE_LIMIT)

    idx_surr = np.random.choice(len(X_clean_remainder), surrogate_size, replace=False)
    X_surr_pool = X_clean_remainder[idx_surr]
    y_surr_pool = y_clean_remainder[idx_surr]

    try:
        X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(
            X_surr_pool, y_surr_pool, test_size=0.2, stratify=y_surr_pool, random_state=seed_val
        )
    except ValueError:
        X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(
            X_surr_pool, y_surr_pool, test_size=0.2, random_state=seed_val
        )

    # --- 3. CUSTOM GRADIENT ASCENT EXECUTION ---
    def train_svm_local(X_t, y_t):
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed_val)
        svm.fit(X_t, y_t)
        return svm

    y_s_val_hinge = np.where(y_s_val == 0, -1, 1)
    malicious_x_list = []
    malicious_y_list = []

    # Get initial points to perturb
    initial_poison_x = np.copy(X_full[idx_poison])
    initial_poison_y = 1 - np.copy(y_full[idx_poison])

    for i in range(len(initial_poison_x)):
        xc = initial_poison_x[i].reshape(1, -1)
        yc = np.array([initial_poison_y[i]])
        
        prev_loss, step_size = None, 0.1
        X_poisoned = np.vstack([X_s_train, xc])
        y_poisoned = np.hstack([y_s_train, yc])
        svm = train_svm_local(X_poisoned, y_poisoned)

        for _ in range(MAX_ITERATIONS):
            gamma_val = svm._gamma
            diffs = xc - svm.support_vectors_
            sq_dists = np.linalg.norm(diffs, axis=1) ** 2
            K = np.exp(-gamma_val * sq_dists).reshape(-1, 1)

            y_poisoned_hinge = np.where(y_poisoned == 0, -1, 1)
            yi_hinge = y_poisoned_hinge[svm.support_].reshape(-1, 1)
            
            abs_alpha_i = np.abs(svm.dual_coef_[0].reshape(-1, 1))
            coeffs = abs_alpha_i * yi_hinge * K * (2 * gamma_val)
            gradient = np.sum(coeffs * diffs, axis=0).reshape(1, -1)

            decision_values = svm.decision_function(X_s_val)
            L_xc = np.sum(np.maximum(0, 1 - y_s_val_hinge * decision_values))

            xc_new = xc + step_size * gradient
            X_poisoned[-1] = xc_new.flatten()
            svm = train_svm_local(X_poisoned, y_poisoned)

            if prev_loss is not None and abs(L_xc - prev_loss) < EPSILON: 
                break
            
            prev_loss = L_xc
            xc = xc_new.copy()
            step_size *= 0.9

        malicious_x_list.append(xc.flatten())
        malicious_y_list.append(yc[0])

    # --- 4. RECOMBINE ---
    malicious_x = np.vstack(malicious_x_list)
    malicious_y = np.array(malicious_y_list)
    
    X_final = np.vstack([X_clean_remainder, malicious_x])
    y_final = np.concatenate([y_clean_remainder, malicious_y])
    
    to_csv(X_final, y_final, cols, path_poison_data)
    return path_poison_data


class PoisSVMPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="poissvm_svm", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X_full, y_full, cols = open_csv(file_path)
        y_full = np.where(y_full == -1, 0, y_full)
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        self.logger.info(f"Dispatching PoisSVM rates {advx_range} in parallel. Dataset size: {len(X_full)}...")

        for rate in advx_range:
            try:
                _generate_poissvm_rate(rate, X_full, y_full, cols, path_output_base)
            except Exception as exc:
                self.logger.error(f'PoisSVM attack failed for rate {rate} on {dataname}: {exc}')

        path_poison_data_list = [f'{path_output_base}_poissvm_svm_{rate:.2f}.csv' for rate in advx_range]

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
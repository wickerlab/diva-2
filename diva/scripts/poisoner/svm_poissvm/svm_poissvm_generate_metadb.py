import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

from ...base_poisoner import BasePoisoner
from ...utils.utils import open_csv, to_csv

RANDOM_SEED = 100
MAX_ITERATIONS = 25
EPSILON = 1e-6

class PoisSVMPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="poissvm_svm", base_folder=base_folder)

    def train_svm(self, X_train, y_train):
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_SEED)
        svm.fit(X_train, y_train)
        return svm

    def initialize_attack_point(self, X_train, y_train, attacked_class=1):
        np.random.seed(RANDOM_SEED)
        class_indices = np.where(y_train == attacked_class)[0]
        svm_temp = self.train_svm(X_train, y_train)
        
        class_support_indices = [i for i in svm_temp.support_ if y_train[i] == attacked_class]
        initial_index = np.random.choice(class_support_indices) if class_support_indices else np.random.choice(class_indices)
        
        return X_train[initial_index].copy(), 1 - y_train[initial_index]

    def gradient_ascent_attack(self, X_train, y_train, X_val, y_val):
        xc, yc = self.initialize_attack_point(X_train, y_train, attacked_class=1)
        xc, yc = xc.reshape(1, -1), np.array([yc])

        # Hinge loss requires labels in {-1, 1}
        y_val_hinge = np.where(y_val == 0, -1, 1)

        prev_loss, step_size = None, 0.1
        X_poisoned, y_poisoned = np.vstack([X_train, xc]), np.hstack([y_train, yc])
        svm = self.train_svm(X_poisoned, y_poisoned)

        for _ in range(MAX_ITERATIONS):
            gamma_val = svm._gamma
            diffs = xc - svm.support_vectors_
            sq_dists = np.linalg.norm(diffs, axis=1) ** 2
            K = np.exp(-gamma_val * sq_dists).reshape(-1, 1)

            # Map current SVM labels to {-1, 1} to prevent zeroing out Class 0
            y_poisoned_hinge = np.where(y_poisoned == 0, -1, 1)
            yi_hinge = y_poisoned_hinge[svm.support_].reshape(-1, 1)
            
            # Scikit-learn's dual_coef_ natively stores (alpha_i * y_i). 
            # To isolate absolute alpha_i as intended by the original heuristic, we take the absolute value.
            abs_alpha_i = np.abs(svm.dual_coef_[0].reshape(-1, 1))
            
            coeffs = abs_alpha_i * yi_hinge * K * (2 * gamma_val)
            gradient = np.sum(coeffs * diffs, axis=0).reshape(1, -1)

            # Evaluate correct Hinge Loss
            decision_values = svm.decision_function(X_val)
            L_xc = np.sum(np.maximum(0, 1 - y_val_hinge * decision_values))

            xc_new = xc + step_size * gradient
            X_poisoned[-1] = xc_new.flatten()
            svm = self.train_svm(X_poisoned, y_poisoned)

            if prev_loss is not None and abs(L_xc - prev_loss) < EPSILON: 
                break
            
            prev_loss = L_xc
            xc = xc_new.copy()
            step_size *= 0.9

        return X_poisoned, y_poisoned, svm

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        y = np.where(y == -1, 0, y)
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        path_poison_data_list = []
        current_poison_count = 0
        
        # Track surrogate state
        X_surr, y_surr = X_train.copy(), y_train.copy()

        for rate in advx_range:
            path_poison_data = f"{path_output_base}_poissvm_svm_{rate:.2f}.csv"
            n_poison = int(len(X) * rate)

            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
                if n_poison > current_poison_count:
                    # Recover state if we skipped a previously generated file
                    X_saved, y_saved, _ = open_csv(path_poison_data)
                    new_x, new_y = X_saved[len(X):], y_saved[len(y):]
                    X_surr = np.vstack([X_train, new_x])
                    y_surr = np.concatenate([y_train, new_y])
                    current_poison_count = n_poison
            else:
                points_to_add = n_poison - current_poison_count
                if points_to_add > 0:
                    self.logger.info(f'     Generating {rate * 100:.0f}% poison data via PoisSVM...')
                    for _ in tqdm(range(points_to_add), ncols=100, desc=f"Poisoning to {rate:.2f}"):
                        X_surr, y_surr, _ = self.gradient_ascent_attack(X_surr, y_surr, X_val, y_val)
                    current_poison_count = n_poison

                if n_poison == 0:
                    to_csv(X, y, cols, path_poison_data)
                else:
                    new_x = X_surr[len(X_train):]
                    new_y = y_surr[len(y_train):]
                    X_final = np.vstack([X, new_x])
                    y_final = np.concatenate([y, new_y])
                    to_csv(X_final, y_final, cols, path_poison_data)

            path_poison_data_list.append(path_poison_data)

        # Meta Database append handling
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
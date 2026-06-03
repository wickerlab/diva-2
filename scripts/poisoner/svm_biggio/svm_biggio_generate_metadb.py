import os
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pathlib import Path

from ...utils.utils import open_csv, to_csv
from ...base_poisoner import BasePoisoner
from .utils.poisoning import PoisoningAttackSVM
from .utils.model import ScikitlearnClassifierSVC

warnings.filterwarnings('ignore')

def _generate_biggio_rate(rate, X_full, y_full, cols, path_output_base):
    path_poison_data = f'{path_output_base}_biggio_svm_{rate:.2f}.csv'
    n_poison = int(len(X_full) * rate) 
    
    if os.path.exists(path_poison_data):
        return path_poison_data
        
    if n_poison == 0:
        to_csv(X_full, y_full, cols, path_poison_data)
        return path_poison_data

    # --- 1. REPLACEMENT STRATEGY: Pick unique points to poison and remove from clean pool ---
    seed_val = (hash(path_output_base) + int(rate * 100)) % (2**32)
    np.random.seed(seed_val)
    
    idx_all = np.arange(len(X_full))
    # Pick n_poison UNIQUE indices to become our attack points
    idx_poison = np.random.choice(idx_all, n_poison, replace=False)
    idx_clean = np.setdiff1d(idx_all, idx_poison)
    
    # The data we keep clean
    X_clean_remainder = X_full[idx_clean]
    y_clean_remainder = y_full[idx_clean]
    
    # The points we will perturb (guaranteed unique starting points)
    initial_poison_x = np.copy(X_full[idx_poison])
    # The attacker wants the model to classify them as the opposite class
    initial_poison_y = 1 - np.copy(y_full[idx_poison]) 
    initial_poison_y_ohe = np.eye(2)[initial_poison_y.astype(int)]

    # --- 2. SURROGATE BACKGROUND: Fast but mathematically valid (Max 1000 points) ---
    MIN_STABLE_BOUNDARY = 200
    MULTIPLIER = 2
    ideal_surrogate_size = max(MIN_STABLE_BOUNDARY, MULTIPLIER * n_poison)
    surrogate_size = min(len(X_clean_remainder), ideal_surrogate_size)
    MAX_COMPUTE_LIMIT = 1000
    if surrogate_size > MAX_COMPUTE_LIMIT:
        surrogate_size = MAX_COMPUTE_LIMIT
    if len(X_clean_remainder) > surrogate_size:
        idx_surr = np.random.choice(len(X_clean_remainder), surrogate_size, replace=False)
        X_surr_pool = X_clean_remainder[idx_surr]
        y_surr_pool = y_clean_remainder[idx_surr]
    else:
        X_surr_pool = X_clean_remainder
        y_surr_pool = y_clean_remainder

    try:
        X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(
            X_surr_pool, y_surr_pool, test_size=0.2, stratify=y_surr_pool, random_state=seed_val
        )
    except ValueError:
        X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(
            X_surr_pool, y_surr_pool, test_size=0.2, random_state=seed_val
        )

    y_s_train_ohe = np.eye(2)[y_s_train.astype(int)]
    y_s_val_ohe = np.eye(2)[y_s_val.astype(int)]
    clip_bounds = (float(np.min(X_s_train)), float(np.max(X_s_train)))

    # --- 3. EXECUTE ATTACK ---
    clf_surrogate = SVC(kernel='linear', C=1.0)
    attack_classifier = ScikitlearnClassifierSVC(model=clf_surrogate, clip_values=clip_bounds, nb_classes=2)
    attack_classifier.fit(X_s_train, y_s_train_ohe)
    
    attack = PoisoningAttackSVM(
        classifier=attack_classifier, step=0.01, eps=1.0,
        x_train=X_s_train, y_train=y_s_train_ohe,
        x_val=X_s_val, y_val=y_s_val_ohe, max_iter=20
    )
    
    malicious_x, malicious_y_ohe = attack.poison(initial_poison_x, initial_poison_y_ohe)
    malicious_y = np.argmax(malicious_y_ohe, axis=1)
    
    # --- 4. RECOMBINE: Final size is EXACTLY the same as original X_full ---
    X_final = np.vstack([X_clean_remainder, malicious_x])
    y_final = np.concatenate([y_clean_remainder, malicious_y])
    
    to_csv(X_final, y_final, cols, path_poison_data)
    
    return path_poison_data

class BiggioSvmPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="biggio_svm", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X_full, y_full, cols = open_csv(file_path)
        y_full = np.where(y_full == -1, 0, y_full)
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        self.logger.info(f"Dispatching Biggio rates {advx_range} in parallel. Dataset size: {len(X_full)}...")
        
        for rate in advx_range:
            try:
                _generate_biggio_rate(rate, X_full, y_full, cols, path_output_base)
            except Exception as exc:
                self.logger.error(f'Biggio attack failed for rate {rate} on {dataname}: {exc}')
                

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
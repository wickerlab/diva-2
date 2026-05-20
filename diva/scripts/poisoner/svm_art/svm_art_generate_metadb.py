import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from art.estimators.classification import SklearnClassifier
from art.attacks.poisoning import PoisoningAttackSVM
from pathlib import Path
import concurrent.futures

from ...utils.utils import open_csv, to_csv
from ...base_poisoner import BasePoisoner

warnings.filterwarnings('ignore')

def _process_art_chunk(chunk_data):
    """
    Isolated worker function for a specific batch. 
    Guarantees unique PRNG state and local surrogate math.
    """
    x_chunk, y_chunk_art, X_clean_remainder, y_clean_remainder, chunk_seed = chunk_data
    
    # Ensure thread-local randomness
    np.random.seed(chunk_seed)
    
    # Each thread randomly samples a unique 50-point background
    MAX_COMPUTE_LIMIT = 50
    surrogate_size = min(len(X_clean_remainder), MAX_COMPUTE_LIMIT)

    idx_surr = np.random.choice(len(X_clean_remainder), surrogate_size, replace=False)
    X_surr_pool = X_clean_remainder[idx_surr]
    y_surr_pool = y_clean_remainder[idx_surr]

    try:
        X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(
            X_surr_pool, y_surr_pool, test_size=0.2, stratify=y_surr_pool, random_state=chunk_seed
        )
    except ValueError:
        X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(
            X_surr_pool, y_surr_pool, test_size=0.2, random_state=chunk_seed
        )

    y_s_train_art = np.eye(2)[y_s_train.astype(int)]
    y_s_val_art = np.eye(2)[y_s_val.astype(int)]
    clip_bounds = (float(np.min(X_s_train)), float(np.max(X_s_train)))

    # Fresh instantiation for every chunk to reset the SVM background safely in threads
    clf_surrogate = SVC(kernel='linear', C=1.0)
    clf_surrogate.fit(X_s_train, y_s_train)
    art_surrogate = SklearnClassifier(model=clf_surrogate, clip_values=clip_bounds)
    
    attack = PoisoningAttackSVM(
        classifier=art_surrogate, step=0.2, eps=1.0,
        x_train=X_s_train, y_train=y_s_train_art, 
        x_val=X_s_val, y_val=y_s_val_art, 
        max_iter=5
    )
    
    chunk_malicious_x, chunk_malicious_y_art = attack.poison(x_chunk, y_chunk_art)
    return chunk_malicious_x, np.argmax(chunk_malicious_y_art, axis=1)


def _generate_art_rate(rate, X_full, y_full, cols, path_output_base):
    path_poison_data = f'{path_output_base}_art_svm_{rate:.2f}.csv'
    n_poison = int(len(X_full) * rate) 
    
    if os.path.exists(path_poison_data):
        return path_poison_data
        
    if n_poison == 0:
        to_csv(X_full, y_full, cols, path_poison_data)
        return path_poison_data

    # --- 1. REPLACEMENT STRATEGY ---
    seed_val = (hash(path_output_base) + int(rate * 100)) % (2**32)
    np.random.seed(seed_val)
    
    idx_all = np.arange(len(X_full))
    idx_poison = np.random.choice(idx_all, n_poison, replace=False)
    idx_clean = np.setdiff1d(idx_all, idx_poison)
    
    X_clean_remainder = X_full[idx_clean]
    y_clean_remainder = y_full[idx_clean]
    
    initial_poison_x = np.copy(X_full[idx_poison])
    initial_poison_y = 1 - np.copy(y_full[idx_poison]) 
    initial_poison_y_art = np.eye(2)[initial_poison_y.astype(int)]

    # --- 2. PARALLEL BATCHED EXECUTION ---
    CHUNK_SIZE = 10
    chunks = []
    
    # Prepare the data chunks for the threads
    for i in range(0, len(initial_poison_x), CHUNK_SIZE):
        x_chunk = initial_poison_x[i:i + CHUNK_SIZE]
        y_chunk_art = initial_poison_y_art[i:i + CHUNK_SIZE]
        
        # Give each chunk a mathematically distinct seed to prevent cloning
        chunk_seed = (seed_val + i) % (2**32)
        
        # We pass the full clean remainder pool so the thread can sample it
        chunks.append((
            x_chunk, y_chunk_art, 
            X_clean_remainder, y_clean_remainder, 
            chunk_seed
        ))

    malicious_x_list = []
    malicious_y_list = []

    # Run the chunks simultaneously using ThreadPool
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(_process_art_chunk, chunks))
        
        for mx, my in results:
            malicious_x_list.append(mx)
            malicious_y_list.append(my)

    # --- 3. RECOMBINE ---
    malicious_x = np.vstack(malicious_x_list)
    malicious_y = np.concatenate(malicious_y_list)
    
    X_final = np.vstack([X_clean_remainder, malicious_x])
    y_final = np.concatenate([y_clean_remainder, malicious_y])
    
    to_csv(X_final, y_final, cols, path_poison_data)
    
    return path_poison_data


class ArtSvmPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="art_svm", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X_full, y_full, cols = open_csv(file_path)
        y_full = np.where(y_full == -1, 0, y_full)
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        self.logger.info(f"Dispatching ART rates {advx_range} in parallel. Dataset size: {len(X_full)}...")

        for rate in advx_range:
            try :
                _generate_art_rate(rate, X_full, y_full, cols, path_output_base)
            except Exception as exc:
                self.logger.error(f'ART attack failed for rate {rate} on {dataname}: {exc}')

        path_poison_data_list = [f'{path_output_base}_art_svm_{rate:.2f}.csv' for rate in advx_range]

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
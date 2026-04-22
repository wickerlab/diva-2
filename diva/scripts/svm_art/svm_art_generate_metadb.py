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

from ..utils.utils import open_csv, to_csv
from ..base_poisoner import BasePoisoner

warnings.filterwarnings('ignore')

class ArtSvmPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="art_svm", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        y = np.where(y == -1, 0, y)
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.complexity_dir, dataname)

        # ART requires a validation set to trace gradients. Standard 80/20 surrogate split.
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        y_val_art = np.eye(2)[y_val.astype(int)]
        clip_bounds = (float(np.min(X_train)), float(np.max(X_train)))

        path_poison_data_list = []
        current_poison_count = 0
        
        # Initialize surrogate state tracking
        X_surr, y_surr = X_train.copy(), y_train.copy()

        for rate in advx_range:
            path_poison_data = f'{path_output_base}_art_svm_{rate:.2f}.csv'
            n_poison = int(len(X_train) * rate)
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
                # Recover cumulative state if we skipped a previously generated file
                if n_poison > current_poison_count:
                    X_saved, y_saved, _ = open_csv(path_poison_data)
                    # Extract only the dynamically added malicious points
                    malicious_x = X_saved[len(X_train):]
                    malicious_y = y_saved[len(y_train):]
                    X_surr = np.vstack([X_train, malicious_x])
                    y_surr = np.concatenate([y_train, malicious_y])
                    current_poison_count = n_poison
            else:
                points_to_add = n_poison - current_poison_count
                
                if points_to_add > 0:
                    self.logger.info(f'     Generating {points_to_add} new points to reach {rate * 100:.0f}% via ART...')
                    
                    # Update surrogate model with the cumulative poisoned state
                    clf_surrogate = SVC(kernel='linear', C=10.0)
                    clf_surrogate.fit(X_surr, y_surr)
                    art_surrogate = SklearnClassifier(model=clf_surrogate, clip_values=clip_bounds)
                    
                    y_surr_art = np.eye(2)[y_surr.astype(int)]
                    
                    attack = PoisoningAttackSVM(
                        classifier=art_surrogate, step=0.2, eps=1.0,
                        x_train=X_surr, y_train=y_surr_art, x_val=X_val, y_val=y_val_art, max_iter=5
                    )
                    
                    # Generate starting coordinates using random clean points
                    idx = np.random.choice(len(X_train), min(points_to_add, len(X_train)), replace=True)
                    initial_poison_x = np.copy(X_train[idx])
                    initial_poison_y = 1 - np.copy(y_train[idx])
                    initial_poison_y_art = np.eye(2)[initial_poison_y.astype(int)]
                    
                    # Execute the attack for the delta
                    new_malicious_x, new_malicious_y_art = attack.poison(initial_poison_x, initial_poison_y_art)
                    new_malicious_y = np.argmax(new_malicious_y_art, axis=1)
                    
                    # Inject into surrogate state
                    X_surr = np.vstack([X_surr, new_malicious_x])
                    y_surr = np.concatenate([y_surr, new_malicious_y])
                    current_poison_count = n_poison
                
                # Save the current state
                if n_poison == 0:
                    to_csv(X_train, y_train, cols, path_poison_data)
                else:
                    to_csv(X_surr, y_surr, cols, path_poison_data)

            path_poison_data_list.append(path_poison_data)

        data = {'Data': np.tile(dataname, reps=len(advx_range)), 'Path.Poison': path_poison_data_list, 'Rate': advx_range}
        pd.DataFrame(data).to_csv(self.csv_score, mode='a' if os.path.exists(self.csv_score) else 'w', header=not os.path.exists(self.csv_score), index=False)
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pathlib import Path

from ..utils.utils import open_csv, to_csv
from ..base_poisoner import BasePoisoner
from .utils.poisoning import PoisoningAttackSVM
from .utils.model import ScikitlearnClassifierSVC

warnings.filterwarnings('ignore')

class BiggioSvmPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="biggio_svm", base_folder=base_folder)

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        
        # Abort processing if dataset is too large to prevent endless hanging
        if len(X) > 500:
            self.logger.info(f"This dataset contains {len(X)} points, more than 500. Skipping for Biggio.")
            return 
            
        y = np.where(y == -1, 0, y)
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.complexity_dir, dataname)

        self.logger.warning("Using specific 20/80 train/val split for surrogate gradients.")
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=42)

        y_val_ohe = np.eye(2)[y_val.astype(int)]
        clip_bounds = (float(np.min(X_train)), float(np.max(X_train)))

        path_poison_data_list = []
        current_poison_count = 0
        
        # Initialize surrogate state tracking
        X_surr, y_surr = X_train.copy(), y_train.copy()

        for rate in advx_range:
            path_poison_data = f'{path_output_base}_biggio_svm_{rate:.2f}.csv'
            n_poison = int(len(X_train) * rate) # Rate relative to total dataset size
            
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
                    self.logger.info(f'  Generating {points_to_add} new points to reach {rate * 100:.0f}% via Biggio...')
                    
                    # Update surrogate model with the cumulative poisoned state
                    clf_surrogate = SVC(kernel='linear', C=1.0)
                    attack_classifier = ScikitlearnClassifierSVC(model=clf_surrogate, clip_values=clip_bounds, nb_classes=2)
                    
                    y_surr_ohe = np.eye(2)[y_surr.astype(int)]
                    attack_classifier.fit(X_surr, y_surr_ohe)
                    
                    attack = PoisoningAttackSVM(
                        classifier=attack_classifier, step=0.01, eps=1.0,
                        x_train=X_surr, y_train=y_surr_ohe,
                        x_val=X_val, y_val=y_val_ohe, max_iter=20
                    )
                    
                    # Generate starting coordinates using random clean points
                    idx = np.random.choice(len(X_train), min(points_to_add, len(X_train)), replace=True)
                    initial_poison_x = np.copy(X_train[idx])
                    initial_poison_y = 1 - np.copy(y_train[idx]) 
                    initial_poison_y_ohe = np.eye(2)[initial_poison_y.astype(int)]
                    
                    # Execute the attack for the delta
                    new_malicious_x, new_malicious_y_ohe = attack.poison(initial_poison_x, initial_poison_y_ohe)
                    new_malicious_y = np.argmax(new_malicious_y_ohe, axis=1)
                    
                    # Inject into surrogate state
                    X_surr = np.vstack([X_surr, new_malicious_x])
                    y_surr = np.concatenate([y_surr, new_malicious_y])
                    current_poison_count = n_poison
                    
                # Save the current state
                if n_poison == 0:
                    # Using X_train to keep the dimensions strictly consistent
                    to_csv(X_train, y_train, cols, path_poison_data)
                else:
                    to_csv(X_surr, y_surr, cols, path_poison_data)

            path_poison_data_list.append(path_poison_data)

        data = {
            'Data': np.tile(dataname, reps=len(advx_range)),
            'Path.Poison': path_poison_data_list,
            'Rate': advx_range
        }
        pd.DataFrame(data).to_csv(self.csv_score, mode='a' if os.path.exists(self.csv_score) else 'w', 
                                  header=not os.path.exists(self.csv_score), index=False)
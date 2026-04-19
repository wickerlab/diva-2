import os
import time
import warnings
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC

# --- Import your BasePoisoner and Utils ---
from ..utils.utils import open_csv, to_csv
from ..base_poisoner import BasePoisoner

# --- Import the custom Biggio implementation ---
from .utils.poisoning import PoisoningAttackSVM
from .utils.model import ScikitlearnClassifierSVC

warnings.filterwarnings('ignore')

class BiggioSvmPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="biggio_svm", base_folder=base_folder)

    def compute_and_save_poisoned_data(self, X_train, y_train, X_val, y_val, X_test, y_test, path_output_base, cols, advx_range):
        # The custom attack requires one-hot encoded labels for train and validation
        num_classes = 2
        y_train_ohe = np.eye(num_classes)[y_train.astype(int)]
        y_val_ohe = np.eye(num_classes)[y_val.astype(int)]
        
        x_min = np.min(X_train)
        x_max = np.max(X_train)
        clip_bounds = (float(x_min), float(x_max))

        # Train initial clean classifier to get baseline accuracy on train and test
        clf_clean = SVC(kernel='linear', C=1.0)
        clf_clean.fit(X_train, y_train)
        acc_train_clean = clf_clean.score(X_train, y_train)
        acc_test_clean = clf_clean.score(X_test, y_test)

        accuracy_train_clean = [acc_train_clean] * len(advx_range)
        accuracy_test_clean = [acc_test_clean] * len(advx_range)
        accuracy_train_poison, accuracy_test_poison, path_poison_data_list = [], [], []

        for rate in advx_range:
            path_poison_data = f'{path_output_base}_biggio_svm_{rate:.2f}.csv'
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Already generated poison data loaded. Skip poisoning.')
                X_poison, y_poison, _ = open_csv(path_poison_data)
            else:
                time_start = time.time()
                n_poison = int(len(X_train) * rate)

                self.logger.info(f'  Generating {rate * 100:.0f}% poison data ({n_poison} points) via Biggio Attack...')
                
                if n_poison == 0:
                    X_poison, y_poison = X_train, y_train
                else:
                    # Initialize the attack wrapper using the ScikitlearnClassifierSVC
                    attack_classifier = ScikitlearnClassifierSVC(
                        model=SVC(kernel='linear', C=1.0),
                        clip_values=clip_bounds,
                        nb_classes=num_classes
                    )
                    attack_classifier.fit(X_train, y_train_ohe)

                    # Initialize your custom attack class, explicitly passing the validation set
                    attack = PoisoningAttackSVM(
                        classifier=attack_classifier,
                        step=0.01,
                        eps=1.0,
                        x_train=X_train,
                        y_train=y_train_ohe,
                        x_val=X_val,
                        y_val=y_val_ohe,
                        max_iter=20
                    )
                    
                    # Generate Attack Points
                    idx = np.random.choice(len(X_train), n_poison, replace=False)
                    initial_poison_x = np.copy(X_train[idx])
                    
                    # Target the opposite class for the attack
                    initial_poison_y = 1 - np.copy(y_train[idx]) 
                    initial_poison_y_ohe = np.eye(num_classes)[initial_poison_y.astype(int)]
                    
                    # Execute the iterative poisoning loop from poisoning.py
                    malicious_x, malicious_y_ohe = attack.poison(initial_poison_x, initial_poison_y_ohe)
                    malicious_y = np.argmax(malicious_y_ohe, axis=1)
                    
                    # Append new points
                    X_poison = np.vstack([X_train, malicious_x])
                    y_poison = np.concatenate([y_train, malicious_y])

                time_elapse = time.time() - time_start
                self.logger.info(f'  Generation took {time_elapse:.1f}s')
                
                to_csv(X_poison, y_poison, cols, path_poison_data)

            # Evaluate on standard sklearn SVC
            clf_poison = SVC(kernel='linear', C=1.0)
            clf_poison.fit(X_poison, y_poison)
            
            acc_train_poison = clf_poison.score(X_poison, y_poison)
            acc_test_poison = clf_poison.score(X_test, y_test)
            
            self.logger.info(f'  P-Rate [{rate * 100:.2f}] Acc P-train: {acc_train_poison * 100:.2f} C-test: {acc_test_poison * 100:.2f}')
            path_poison_data_list.append(path_poison_data)
            accuracy_train_poison.append(acc_train_poison)
            accuracy_test_poison.append(acc_test_poison)
        
        return accuracy_train_clean, accuracy_test_clean, accuracy_train_poison, accuracy_test_poison, path_poison_data_list

    def apply_poisoning(self, file_path, advx_range):
        X_raw, y_raw, cols = open_csv(file_path)
        
        # Sanitize labels to {0, 1}
        y_raw = np.where(y_raw == -1, 0, y_raw)

        # Separate data by class
        target_digit1_xdata = X_raw[y_raw == 0]
        target_digit1_ydata = y_raw[y_raw == 0]
        
        target_digit2_xdata = X_raw[y_raw == 1]
        target_digit2_ydata = y_raw[y_raw == 1]

        # Determine limits based on the minority class to ensure balanced split
        min_samples = min(len(target_digit1_xdata), len(target_digit2_xdata))

        # Target: 50 train, 200 val, 1000 test per class (1250 total per class required)
        if min_samples >= 1250:
            tr_end, val_end, te_end = 50, 250, 1250
        else:
            # Fallback proportional split if dataset is too small (~4% Train, ~16% Val, ~80% Test)
            tr_end = int(min_samples * 0.04)
            val_end = tr_end + int(min_samples * 0.16)
            te_end = min_samples
            
            # Safety constraints in case of tiny datasets
            if tr_end == 0: tr_end = 1
            if val_end <= tr_end: val_end = tr_end + 1
            
            self.logger.info(f"Dataset too small for exact split. Using dynamic split: "
                             f"Train:{tr_end}/class, Val:{val_end-tr_end}/class, Test:{te_end-val_end}/class.")

        # Slicing datasets
        X_train = np.concatenate([target_digit1_xdata[:tr_end], target_digit2_xdata[:tr_end]], axis=0)
        y_train = np.concatenate([target_digit1_ydata[:tr_end], target_digit2_ydata[:tr_end]], axis=0)

        X_val = np.concatenate([target_digit1_xdata[tr_end:val_end], target_digit2_xdata[tr_end:val_end]], axis=0)
        y_val = np.concatenate([target_digit1_ydata[tr_end:val_end], target_digit2_ydata[tr_end:val_end]], axis=0)

        X_test = np.concatenate([target_digit1_xdata[val_end:te_end], target_digit2_xdata[val_end:te_end]], axis=0)
        y_test = np.concatenate([target_digit1_ydata[val_end:te_end], target_digit2_ydata[val_end:te_end]], axis=0)

        # Shuffle permutations
        perm_train = np.random.permutation(X_train.shape[0])
        perm_val = np.random.permutation(X_val.shape[0])
        perm_test = np.random.permutation(X_test.shape[0])

        X_train, y_train = X_train[perm_train], y_train[perm_train]
        X_val, y_val = X_val[perm_val], y_val[perm_val]
        X_test, y_test = X_test[perm_test], y_test[perm_test]

        dataname = Path(file_path).stem
        output_base_path = os.path.join(self.complexity_dir, dataname)

        # Pass X_val and y_val explicitly
        acc_train_c, acc_test_c, acc_train_p, acc_test_p, path_poison_list = self.compute_and_save_poisoned_data(
            X_train, y_train, X_val, y_val, X_test, y_test, output_base_path, cols, advx_range
        )

        data = {
            'Data': np.tile(dataname, reps=len(advx_range)),
            'Path.Poison': path_poison_list,
            'Rate': advx_range,
            'Train.Clean': acc_train_c,
            'Test.Clean': acc_test_c,
            'Train.Poison': acc_train_p,
            'Test.Poison': acc_test_p,
        }
        df = pd.DataFrame(data)
        
        df.to_csv(self.csv_score, mode='a' if os.path.exists(self.csv_score) else 'w', 
                    header=not os.path.exists(self.csv_score), index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", default="data", type=str, help="The output folder.")
    parser.add_argument("-s", "--step", type=float, default=0.05, help="Spacing between poisoning rates.")
    parser.add_argument("-m", "--max", type=float, default=0.41, help="End of interval for poisoning rates.")
    parser.add_argument(
        "-e", "--entrypoint", type=str,
        default="poison", help="Entrypoint for the pipeline.",
        choices=["poison", "cmeasure","metadb"])
    args = parser.parse_args()

    base = args.folder
    os.makedirs(base, exist_ok=True)
    advx_range = np.arange(0, args.max, args.step)

    poisoners = [BiggioSvmPoisoner(base_folder=base)]
    files = [f for f in Path(f'{base}/clean_data/').iterdir() if f.is_file()]

    for poisoner in poisoners:
        poisoner.run_pipeline(files, advx_range, entrypoint=args.entrypoint)
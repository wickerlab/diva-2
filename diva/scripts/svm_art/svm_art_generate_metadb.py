import os
import time
import warnings
from pathlib import Path
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from art.estimators.classification import SklearnClassifier
from art.attacks.poisoning import PoisoningAttackSVM

from ..utils.utils import open_csv, to_csv
from ..base_poisoner import BasePoisoner

warnings.filterwarnings('ignore')

class ArtSvmPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="art_svm", base_folder=base_folder)

    def compute_and_save_poisoned_data(self, X_train, y_train, X_test, y_test, clf, path_output_base, cols, advx_range):
        
        # ---Convert Scikit-Learn 1D labels to ART One-Hot encoded labels ---
        num_classes = 2
        y_train_art = np.eye(num_classes)[y_train.astype(int)]
        y_test_art = np.eye(num_classes)[y_test.astype(int)]
        
        x_min = np.min(X_train)
        x_max = np.max(X_train)
        clip_bounds = (float(x_min), float(x_max))

        val_subset_size = min(len(X_test), 200) # Use max 200 points to evaluate the attack
        idx_val = np.random.choice(len(X_test), val_subset_size, replace=False)
        
        x_val_subset = X_test[idx_val]
        y_val_subset_art = y_test_art[idx_val]

        acc_train_clean = clf.score(X_train, y_train)
        acc_test_clean = clf.score(X_test, y_test)

        accuracy_train_clean = [acc_train_clean] * len(advx_range)
        accuracy_test_clean = [acc_test_clean] * len(advx_range)
        accuracy_train_poison, accuracy_test_poison, path_poison_data_list = [], [], []

        for rate in advx_range:
            path_poison_data = f'{path_output_base}_art_svm_{rate:.2f}.csv'
            
            # If we already generated this dataset, load it
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Already generated poison data loaded. Skip poisoning.')
                X_poison, y_poison, _ = open_csv(path_poison_data)
            else:
                time_start = time.time()
                
                n_poison = int(len(X_train) * 0.5 *rate)

                self.logger.info(f'  Generating {rate * 0.5 * 100:.0f}% poison data (ie {n_poison} points) via ART...')
                
                if n_poison == 0:
                    X_poison, y_poison = X_train, y_train
                else:
                    surrogate_size = min(len(X_train), 1500) 
                    idx_surrogate = np.random.choice(len(X_train), surrogate_size, replace=False)
                    
                    x_train_surrogate = X_train[idx_surrogate]
                    y_train_surrogate_art = y_train_art[idx_surrogate]

                    # Train the lightweight surrogate
                    clf_surrogate = SVC(kernel='linear')
                    clf_surrogate.fit(x_train_surrogate, np.argmax(y_train_surrogate_art, axis=1))
                    art_classifier_surrogate = SklearnClassifier(model=clf_surrogate, clip_values=clip_bounds)

                    # Initialize the attack strictly on the fast surrogate model
                    attack = PoisoningAttackSVM(
                        classifier=art_classifier_surrogate,
                        step=0.2,
                        eps=1.0,
                        x_train=x_train_surrogate,
                        y_train=y_train_surrogate_art,
                        x_val=x_val_subset,
                        y_val=y_val_subset_art,
                        max_iter=5
                    )
                    
                    # --- Generate Attack Points ---
                    # Select the points from the FULL dataset to turn into poison
                    idx = np.random.choice(len(X_train), n_poison, replace=False)
                    initial_poison_x = np.copy(X_train[idx])
                    
                    # Flip labels and one-hot encode
                    initial_poison_y = 1 - np.copy(y_train[idx]) 
                    initial_poison_y_art = np.eye(num_classes)[initial_poison_y.astype(int)]
                    
                    # Execute the attack (this will now be hundreds of times faster)
                    malicious_x, malicious_y_art = attack.poison(initial_poison_x, initial_poison_y_art)
                    malicious_y = np.argmax(malicious_y_art, axis=1)
                    
                    # INJECTION: Append the new malicious points to the FULL original training data
                    X_poison = np.vstack([X_train, malicious_x])
                    y_poison = np.concatenate([y_train, malicious_y])

                time_elapse = time.time() - time_start
                self.logger.info(f'  Generation took {time_elapse:.1f}s')
                
                # Save the newly injected dataset
                to_csv(X_poison, y_poison, cols, path_poison_data)

            clf_poison = SVC(**clf.get_params())
            clf_poison.fit(X_poison, y_poison)
            
            acc_train_poison = clf_poison.score(X_poison, y_poison)
            acc_test_poison = clf_poison.score(X_test, y_test)
            
            self.logger.info(f'  P-Rate [{rate * 100:.2f}] Acc P-train: {acc_train_poison * 100:.2f} C-test: {acc_test_poison * 100:.2f}')
            path_poison_data_list.append(path_poison_data)
            accuracy_train_poison.append(acc_train_poison)
            accuracy_test_poison.append(acc_test_poison)
        
        return accuracy_train_clean, accuracy_test_clean, accuracy_train_poison, accuracy_test_poison, path_poison_data_list
    def apply_poisoning(self, file_path, advx_range):
            
        X_train, y_train, cols = open_csv(file_path)
        
        # Sanitize labels to {0, 1}
        y_train = np.where(y_train == -1, 0, y_train)
        
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
        dataname = Path(file_path).stem

        # Train the initial clean model required by ART
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)

        output_base_path = os.path.join(self.complexity_dir, dataname)

        acc_train_clean, acc_test_clean, acc_train_poison, acc_test_poison, path_poison_data_list = self.compute_and_save_poisoned_data(
            X_train, y_train, X_test, y_test, clf, output_base_path, cols, advx_range
        )

        data = {
            'Data': np.tile(dataname, reps=len(advx_range)),
            'Path.Poison': path_poison_data_list,
            'Rate': advx_range,
            'Train.Clean': acc_train_clean,
            'Test.Clean': acc_test_clean,
            'Train.Poison': acc_train_poison,
            'Test.Poison': acc_test_poison,
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
        choices= ["poison", "cmeasure","metadb"])
    args = parser.parse_args()

    base = args.folder
    os.makedirs(base, exist_ok=True)
    advx_range = np.arange(0, args.max, args.step)

    # Initialize all your poisoners
    poisoners = [
        ArtSvmPoisoner(base_folder=base)
    ]

    files = [f for f in Path(f'{base}/clean_data/').iterdir() if f.is_file()]

    # Run the standardized pipeline for each method
    for poisoner in poisoners:
        poisoner.run_pipeline(files, advx_range, entrypoint=args.entrypoint)
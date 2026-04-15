import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from scipy.stats import loguniform

from ..utils.utils import create_dir, open_csv, to_csv
from ..base_poisoner import BasePoisoner

import argparse
from pathlib import Path
import logging

warnings.filterwarnings("ignore")

N_ITER_SEARCH = 50
SVM_PARAM_DICT = {
    "C": loguniform(0.01, 10),
    "gamma": loguniform(0.01, 10),
    "kernel": ["rbf"],
}

class FeatureNoisePoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="feature_noise_svm", base_folder=base_folder)

    def inject_feature_noise(self, X_train, rate, noise_level=3.0):
        X_noisy = X_train.copy()
        n_noisy = int(len(X_train) * rate)
        noisy_indices = np.random.choice(len(X_train), size=n_noisy, replace=False)
        noise = np.random.normal(0, noise_level, size=X_train.shape)
        X_noisy[noisy_indices] += noise[noisy_indices]
        return X_noisy

    def compute_and_save_noisy_data(self, X_train, y_train, X_test, y_test, clf, path_output_base, cols, noise_rate_range):
        acc_train_clean = clf.score(X_train, y_train)
        acc_test_clean = clf.score(X_test, y_test)

        accuracy_train_clean = [acc_train_clean] * len(noise_rate_range)
        accuracy_test_clean = [acc_test_clean] * len(noise_rate_range)
        accuracy_train_noisy, accuracy_test_noisy, path_noisy_data_list = [], [], []

        for rate in noise_rate_range:
            path_noisy_data = "{}_featurenoiseinjection_svm_{:.2f}.csv".format(path_output_base, np.round(rate, 2))
            try:
                if os.path.exists(path_noisy_data):
                    self.logger.info(f'     Already generated poison data loaded. Skip poisoning.')
                    X_train_noisy, y_train_noisy, _ = open_csv(path_noisy_data)
                else:
                    X_train_noisy = self.inject_feature_noise(X_train, rate)
                    y_train_noisy = y_train.copy()
                    to_csv(X_train_noisy, y_train_noisy, cols, path_noisy_data)
                
                svm_params = clf.get_params()
                clf_noisy = SVC(**svm_params)
                clf_noisy.fit(X_train_noisy, y_train_noisy)
                acc_train_noisy = clf_noisy.score(X_train_noisy, y_train_noisy)
                acc_test_noisy = clf_noisy.score(X_test, y_test)
            except Exception as e:
                self.logger.error(e)
                acc_train_noisy, acc_test_noisy = 0, 0
            
            self.logger.info("      Noise Rate [{:.2f}%] - Acc  Noisy Train: {:.2f}%  Test Set: {:.2f}%".format(
                rate * 100, acc_train_noisy * 100, acc_test_noisy * 100))
            
            path_noisy_data_list.append(path_noisy_data)
            accuracy_train_noisy.append(acc_train_noisy)
            accuracy_test_noisy.append(acc_test_noisy)

        return accuracy_train_clean, accuracy_test_clean, accuracy_train_noisy, accuracy_test_noisy, path_noisy_data_list

    def apply_poisoning(self, file_path, advx_range):
        X_train, y_train, cols = open_csv(file_path)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
        dataname = Path(file_path).stem

        clf = SVC()
        random_search = RandomizedSearchCV(clf, param_distributions=SVM_PARAM_DICT, n_iter=N_ITER_SEARCH, cv=5, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        clf = SVC(**random_search.best_params_)
        clf.fit(X_train, y_train)

        acc_train_clean, acc_test_clean, acc_train_noisy, acc_test_noisy, path_noisy_data_list = self.compute_and_save_noisy_data(
            X_train, y_train, X_test, y_test, clf,
            os.path.join(self.complexity_dir, dataname),
            cols, advx_range
        )

        data = {
            "Data": np.tile(dataname, reps=len(advx_range)),
            "Path.Poison": path_noisy_data_list,
            "Rate": advx_range,
            "Train.Clean": acc_train_clean,
            "Test.Clean": acc_test_clean,
            "Train.Poison": acc_train_noisy,
            "Test.Poison": acc_test_noisy,
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
        FeatureNoisePoisoner(base_folder=base)
    ]

    files = [f for f in Path(f'{base}/clean_data/').iterdir() if f.is_file()]

    # Run the standardized pipeline for each method
    for poisoner in poisoners:
        poisoner.run_pipeline(files, advx_range, entrypoint=args.entrypoint)
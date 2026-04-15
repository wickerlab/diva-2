import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from scipy.stats import loguniform

from .utils.alfa import alfa
from ..utils.utils import create_dir, open_csv, to_csv, transform_label
from ..base_poisoner import BasePoisoner
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

import argparse
from pathlib import Path
import logging

warnings.filterwarnings('ignore')

ALFA_MAX_ITER = 5
N_ITER_SEARCH = 10
SVM_PARAM_DICT = {
    'C': loguniform(0.01, 10),
    'gamma': loguniform(0.01, 10),
    'kernel': ['rbf'],
}

class AlfaPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="alfa_svm", base_folder=base_folder)

    def get_y_flip(self, X_train, y_train, rate, svc):
        if rate == 0:
            return y_train
        y_train = transform_label(y_train, target=-1)
        y_flip = alfa(X_train, y_train, rate, svc_params=svc.get_params(), max_iter=ALFA_MAX_ITER)
        y_flip = transform_label(y_flip, target=0)
        return y_flip

    def compute_and_save_flipped_data(self, X_train, y_train, X_test, y_test, clf, path_output_base, cols, advx_range):
        acc_train_clean = clf.score(X_train, y_train)
        acc_test_clean = clf.score(X_test, y_test)

        accuracy_train_clean = [acc_train_clean] * len(advx_range)
        accuracy_test_clean = [acc_test_clean] * len(advx_range)
        accuracy_train_poison, accuracy_test_poison, path_poison_data_list = [], [], []

        for rate in advx_range:
            path_poison_data = '{}_alfa_svm_{:.2f}.csv'.format(path_output_base, np.round(rate, 2))
            try:
                if os.path.exists(path_poison_data):
                    self.logger.info(f'     Already generated poison data loaded. Skip poisoning.')
                    X_train, y_flip, _ = open_csv(path_poison_data)
                else:
                    time_start = time.time()
                    y_flip = self.get_y_flip(X_train, y_train, rate, clf)
                    time_elapse = time.time() - time_start
                    self.logger.info('  Generating {:.0f}% poison labels took {:.1f}s'.format(rate * 100, time_elapse))
                    to_csv(X_train, y_flip, cols, path_poison_data)
                
                svm_params = clf.get_params()
                clf_poison = SVC(**svm_params)
                clf_poison.fit(X_train, y_flip)
                acc_train_poison = clf_poison.score(X_train, y_flip)
                acc_test_poison = clf_poison.score(X_test, y_test)
            except Exception as e:
                self.logger.error(e)
                acc_train_poison, acc_test_poison = 0, 0
            
            self.logger.info('  P-Rate [{:.2f}] Acc  P-train: {:.2f} C-test: {:.2f}'.format(rate * 100, acc_train_poison * 100, acc_test_poison * 100))
            path_poison_data_list.append(path_poison_data)
            accuracy_train_poison.append(acc_train_poison)
            accuracy_test_poison.append(acc_test_poison)
        
        return accuracy_train_clean, accuracy_test_clean, accuracy_train_poison, accuracy_test_poison, path_poison_data_list

    def apply_poisoning(self, file_path, advx_range):
        X_train, y_train, cols = open_csv(file_path)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
        dataname = Path(file_path).stem

        clf = SVC()
        halving_search = HalvingRandomSearchCV(
            clf, 
            param_distributions=SVM_PARAM_DICT, 
            n_candidates='exhaust', 
            factor=3, 
            n_jobs=-1
        )
        halving_search.fit(X_train, y_train)
        
        clf = SVC(**halving_search.best_params_)
        clf.fit(X_train, y_train)

        acc_train_clean, acc_test_clean, acc_train_poison, acc_test_poison, path_poison_data_list = self.compute_and_save_flipped_data(
            X_train, y_train, X_test, y_test, clf,
            os.path.join(self.complexity_dir, dataname),
            cols, advx_range
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
        AlfaPoisoner(base_folder=base)
    ]

    files = [f for f in Path(f'{base}/clean_data/').iterdir() if f.is_file()]

    # Run the standardized pipeline for each method
    for poisoner in poisoners:
        poisoner.run_pipeline(files, advx_range, entrypoint=args.entrypoint)
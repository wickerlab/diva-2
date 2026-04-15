import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

from ..base_poisoner import BasePoisoner
from ..utils.utils import open_csv
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

# Constants
RANDOM_SEED = 100
PCA_COMPONENTS = 10
N_NEIGHBORS = 5
MAX_ITERATIONS = 25
EPSILON = 1e-6

class PoisSVMPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="poissvm_svm", base_folder=base_folder, custom_complexity_dir="numerical_gradient")

    def extract_key(self, filename):
        filename = os.path.basename(filename)
        return "_".join(filename.split("_")[:-1]) 

    def load_and_preprocess_data(self, file):
        df = pd.read_csv(file)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        normalizer = Normalizer()
        X_normalized = normalizer.fit_transform(X_scaled)
        pca = PCA(n_components=min(PCA_COMPONENTS, X_normalized.shape[1]), random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_normalized)

        graph = kneighbors_graph(X_pca, N_NEIGHBORS, mode="connectivity", include_self=True)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        return X_pca, y, labels, n_components

    def train_svm(self, X_train, y_train, C=1.0, gamma="scale"):
        svm = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, class_weight="balanced", random_state=RANDOM_SEED, cache_size=1000)
        svm.fit(X_train, y_train)
        return svm

    def evaluate(self, svm, X_test, y_test):
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        return accuracy, balanced_acc, precision, recall, f1

    def initialize_attack_point(self, X_train, y_train, attacked_class=1):
        np.random.seed(RANDOM_SEED)
        class_indices = np.where(y_train == attacked_class)[0]
        if len(class_indices) == 0:
            raise ValueError(f"No samples found for class {attacked_class}")
        
        svm_temp = self.train_svm(X_train, y_train)
        support_indices = svm_temp.support_
        class_support_indices = [i for i in support_indices if y_train[i] == attacked_class]
        
        initial_index = np.random.choice(class_support_indices) if class_support_indices else np.random.choice(class_indices)
        return X_train[initial_index].copy(), -y_train[initial_index]

    def gradient_ascent_attack(self, X_train, y_train, X_val, y_val, attack_class=1, C=1.0, gamma="scale"):
        xc, yc = self.initialize_attack_point(X_train, y_train, attacked_class=attack_class)
        xc = xc.reshape(1, -1)
        yc = np.array([yc])

        prev_loss = None
        step_size = 0.1
        X_poisoned = np.vstack([X_train, xc])
        y_poisoned = np.hstack([y_train, yc])
        svm = self.train_svm(X_poisoned, y_poisoned, C=C, gamma=gamma)

        for iteration in range(MAX_ITERATIONS):
            dual_coefs = svm.dual_coef_[0]
            support_vectors = svm.support_vectors_
            support_indices = svm.support_
            gamma_value = svm._gamma if gamma in ["scale", "auto"] else gamma

            diffs = xc - support_vectors  # Broadcast subtraction: shape (N_support, N_features)
            sq_dists = np.linalg.norm(diffs, axis=1) ** 2
            K = np.exp(-gamma_value * sq_dists).reshape(-1, 1)

            yi = y_poisoned[support_indices].reshape(-1, 1)
            alpha_i = dual_coefs.reshape(-1, 1)
            
            # Compute coefficients for all support vectors at once
            coeffs = alpha_i * yi * K * (2 * gamma_value)
            
            # Multiply coefficients by differences and sum over axis 0 to get final gradient
            gradient = np.sum(coeffs * diffs, axis=0).reshape(1, -1)

            decision_values = svm.decision_function(X_val)
            hinge_losses = np.maximum(0, 1 - y_val * decision_values)
            L_xc = np.sum(hinge_losses)

            xc_new = xc + step_size * gradient
            X_poisoned[-1] = xc_new.flatten()
            svm = self.train_svm(X_poisoned, y_poisoned, C=C, gamma=gamma)

            if prev_loss is not None and abs(L_xc - prev_loss) < EPSILON:
                break

            prev_loss = L_xc
            xc = xc_new.copy()
            X_poisoned[-1] = xc.flatten()
            step_size *= 0.9

        return X_poisoned, y_poisoned, svm

    def apply_poisoning(self, file, advx_range):
        X, y, _, _ = self.load_and_preprocess_data(file)
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train_full)
        
        base_file_name = "_".join(os.path.basename(file).split("_")[:-1])
        results = []

        # 0% Clean Attack baseline
        svm_clean = self.train_svm(X_train, y_train)
        acc_train_clean, _, _, _, _ = self.evaluate(svm_clean, X_train, y_train)
        acc_test_clean, _, _, _, _ = self.evaluate(svm_clean, X_test, y_test)

        poison_file_name = f"{base_file_name}_numericalgradient_svm_0.00.csv"
        poison_data_path = os.path.join(self.complexity_dir, poison_file_name)
        
        pd.DataFrame(np.hstack([X_train, y_train.reshape(-1, 1)]), 
                        columns=[f"feature_{i}" for i in range(X_train.shape[1])] + ["label"]).to_csv(poison_data_path, index=False)
        
        results.append({
            "Data": os.path.basename(file), "Path.Poison": poison_data_path, "Rate": 0.00,
            "Train.Clean": acc_train_clean, "Test.Clean": acc_test_clean,
            "Train.Poison": acc_train_clean, "Test.Poison": acc_test_clean,
        })

        current_poison_count = 0
        X_poisoned, y_poisoned = X_train.copy(), y_train.copy()
        svm_poisoned = svm_clean

        for rate in advx_range:
            if rate == 0.0: continue
            
            poison_file_name = f"{base_file_name}_numericalgradient_svm_{rate:.2f}.csv"
            poison_data_path = os.path.join(self.complexity_dir, poison_file_name)

            if os.path.exists(poison_data_path):
                self.logger.info(f'     Already generated poison data loaded. Skip poisoning.')
                X_poisoned, y_poisoned, _ = open_csv(poison_data_path)
                svm_poisoned = self.train_svm(X_poisoned, y_poisoned, C=1, gamma="scale")

            else:
                target_num_attack_points = max(1, int(rate * len(X_train)))
                points_to_add = target_num_attack_points - current_poison_count

                # Only generate new points if necessary
                if points_to_add > 0:
                    pbar = tqdm(range(points_to_add), ncols=100, desc=f"Poisoning to {rate:.2f}")
                    for _ in pbar:
                        X_poisoned, y_poisoned, svm_poisoned = self.gradient_ascent_attack(X_poisoned, y_poisoned, X_val, y_val)
                    current_poison_count = target_num_attack_points

            acc_train_poison, _, _, _, _ = self.evaluate(svm_poisoned, X_poisoned, y_poisoned)
            acc_test_poison, _, _, _, _ = self.evaluate(svm_poisoned, X_test, y_test)

            
            pd.DataFrame(np.hstack([X_poisoned, y_poisoned.reshape(-1, 1)]), 
                            columns=[f"feature_{i}" for i in range(X_poisoned.shape[1])] + ["label"]).to_csv(poison_data_path, index=False)

            results.append({
                "Data": os.path.basename(file), "Path.Poison": poison_data_path, "Rate": rate,
                "Train.Clean": acc_train_clean, "Test.Clean": acc_test_clean,
                "Train.Poison": acc_train_poison, "Test.Poison": acc_test_poison,
            })

        df_results = pd.DataFrame(results)
        df_results.to_csv(self.csv_score, mode='a' if os.path.exists(self.csv_score) else 'w', 
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
        PoisSVMPoisoner(base_folder=base)
    ]

    files = [f for f in Path(f'{base}/clean_data/').iterdir() if f.is_file()]

    # Run the standardized pipeline for each method
    for poisoner in poisoners:
        poisoner.run_pipeline(files, advx_range, entrypoint=args.entrypoint)
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Ensure these imports point to the correct locations in your project structure
from .utils.alfa_nn_v3 import get_dual_loss, solveLPNN
from .utils.simple_nn_model import SimpleModel
from .utils.torch_utils import evaluate, train_model
from .utils.utils import create_dir, open_csv, to_csv
from ..base_poisoner import BasePoisoner
import argparse
from pathlib import Path
import logging

warnings.filterwarnings('ignore')

# Training Constants
BATCH_SIZE = 256
HIDDEN_LAYER = 128
LR = 0.01
MAX_EPOCHS = 300
MOMENTUM = 0.9
ALFA_MAX_ITER = 3


class FalfaNNPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="falfa_nn", base_folder=base_folder)
        # Set device once during initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Initialized FALFA NN Poisoner using device: {self.device}")

    def numpy2dataloader(self, X, y, batch_size=BATCH_SIZE, shuffle=True):
        dataset = TensorDataset(
            torch.from_numpy(X).type(torch.float32),
            torch.from_numpy(y).type(torch.int64)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_y_flip(self, model, X_train, y_train, eps, optimizer, loss_fn):
        if eps == 0:
            return y_train

        X_train_tensor = torch.from_numpy(X_train).type(torch.float32)
        tau = get_dual_loss(model, X_train_tensor, self.device)
        alpha = np.zeros_like(tau)
        y_poison = np.copy(y_train).astype(int)

        pbar = tqdm(range(ALFA_MAX_ITER), ncols=100)
        for step in pbar:
            y_poison_next, msg = solveLPNN(alpha, tau, y_true=y_train, eps=eps)
            y_poison_next = np.round(y_poison_next).astype(int)
            pbar.set_postfix({'Optimizer': msg})

            if step > 1 and np.all(y_poison_next == y_poison):
                self.logger.info('  Poison labels converged. Breaking.')
                break
            y_poison = y_poison_next

            # Update model with newly poisoned labels
            dataloader = self.numpy2dataloader(X_train, y_poison)
            train_model(model, dataloader, optimizer=optimizer, loss_fn=loss_fn,
                        device=self.device, max_epochs=MAX_EPOCHS)
            
            # Recalculate dual loss for the next iteration
            alpha = get_dual_loss(model, X_train_tensor, self.device)

        return y_poison

    def compute_and_save_flipped_data(self, X_train, y_train, X_test, y_test, model, path_output_base, cols, advx_range):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

        # Pre-calculate clean evaluation baselines
        dataloader_train_clean = self.numpy2dataloader(X_train, y_train, shuffle=False)
        dataloader_test_clean = self.numpy2dataloader(X_test, y_test, shuffle=False)
        acc_train_clean, _ = evaluate(dataloader_train_clean, model, loss_fn, self.device)
        acc_test_clean, _ = evaluate(dataloader_test_clean, model, loss_fn, self.device)

        accuracy_train_clean = [acc_train_clean] * len(advx_range)
        accuracy_test_clean = [acc_test_clean] * len(advx_range)
        accuracy_train_poison, accuracy_test_poison, path_poison_data_list = [], [], []

        for rate in advx_range:
            path_poison_data = f'{path_output_base}_falfa_nn_{rate:.2f}.csv'
            try:
                # 1. Generate or load poisoned data
                if os.path.exists(path_poison_data):
                    self.logger.info(f'     Already generated poison data loaded. Skip poisoning.')
                    X_train, y_flip, _ = open_csv(path_poison_data)
                else:
                    time_start = time.time()
                    y_flip = self.get_y_flip(model, X_train, y_train, rate, optimizer, loss_fn)
                    time_elapse = time.time() - time_start
                    self.logger.info(f'  Generating {rate * 100:.0f}% poison labels took {time_elapse:.1f}s')
                    to_csv(X_train, y_flip, cols, path_poison_data)
                
                # 2. Train a fresh model on the poisoned data
                n_features = X_train.shape[1]
                poisoned_model = SimpleModel(n_features, hidden_dim=HIDDEN_LAYER, output_dim=2).to(self.device)
                optimizer_poison = torch.optim.SGD(poisoned_model.parameters(), lr=LR, momentum=MOMENTUM)

                dataloader_poison = self.numpy2dataloader(X_train, y_flip)
                train_model(poisoned_model, dataloader_poison, optimizer_poison, loss_fn, self.device, MAX_EPOCHS)

                # 3. Evaluate the poisoned model
                acc_train_poison, _ = evaluate(dataloader_poison, poisoned_model, loss_fn, self.device)
                acc_test_poison, _ = evaluate(dataloader_test_clean, poisoned_model, loss_fn, self.device)

            except Exception as e:
                self.logger.error(f"Error during rate {rate}: {e}")
                acc_train_poison, acc_test_poison = 0, 0
            
            self.logger.info(f'  P-Rate [{rate * 100:.2f}] Acc  P-train: {acc_train_poison * 100:.2f} C-test: {acc_test_poison * 100:.2f}')
            path_poison_data_list.append(path_poison_data)
            accuracy_train_poison.append(acc_train_poison)
            accuracy_test_poison.append(acc_test_poison)
        
        return accuracy_train_clean, accuracy_test_clean, accuracy_train_poison, accuracy_test_poison, path_poison_data_list

    def apply_poisoning(self, file_path, advx_range):
        # Load and split data
        X_train, y_train, cols = open_csv(file_path)
        y_train = np.where(y_train == -1, 0, y_train)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
        dataname = Path(file_path).stem

        # Setup initial clean model
        n_features = X_train.shape[1]
        model = SimpleModel(n_features, hidden_dim=HIDDEN_LAYER, output_dim=2).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
        loss_fn = nn.CrossEntropyLoss()

        # Train the initial clean model
        dataloader_train = self.numpy2dataloader(X_train, y_train)
        train_model(model, dataloader_train, optimizer, loss_fn, self.device, MAX_EPOCHS)

        # Get target paths and run pipeline
        output_base_path = os.path.join(self.complexity_dir, dataname)

        acc_train_clean, acc_test_clean, acc_train_poison, acc_test_poison, path_poison_data_list = self.compute_and_save_flipped_data(
            X_train, y_train, X_test, y_test, model, output_base_path, cols, advx_range
        )

        # Save the final evaluation metrics to the shared CSV
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
        
        # Append to csv_score gracefully
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
        FalfaNNPoisoner(base_folder=base)
    ]

    files = [f for f in Path(f'{base}/clean_data/').iterdir() if f.is_file()]

    # Run the standardized pipeline for each method
    for poisoner in poisoners:
        poisoner.run_pipeline(files, advx_range, entrypoint=args.entrypoint)
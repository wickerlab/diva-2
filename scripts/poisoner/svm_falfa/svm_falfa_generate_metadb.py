import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
from pathlib import Path
import logging

from .utils.alfa_nn_v3 import get_dual_loss, solveLPNN
from .utils.simple_nn_model import SimpleModel
from .utils.torch_utils import train_model
from ...utils.utils import open_csv, to_csv
from ...base_poisoner import BasePoisoner

warnings.filterwarnings('ignore')

BATCH_SIZE = 256
HIDDEN_LAYER = 128
LR = 0.01
MAX_EPOCHS = 300
MOMENTUM = 0.9
ALFA_MAX_ITER = 3

class FalfaNNPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="falfa_nn", base_folder=base_folder)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Initialized FALFA NN Poisoner using device: {self.device}")

    def numpy2dataloader(self, X, y, batch_size=BATCH_SIZE, shuffle=True):
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float32), torch.from_numpy(y).type(torch.int64))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_y_flip(self, model, X_train, y_train, eps, optimizer, loss_fn):
        if eps == 0: return y_train

        X_train_tensor = torch.from_numpy(X_train).type(torch.float32)
        tau = get_dual_loss(model, X_train_tensor, self.device)
        alpha = np.zeros_like(tau)
        y_poison = np.copy(y_train).astype(int)

        pbar = tqdm(range(ALFA_MAX_ITER), ncols=100, desc="Optimizing Labels")
        for step in pbar:
            y_poison_next, msg = solveLPNN(alpha, tau, y_true=y_train, eps=eps)
            y_poison_next = np.round(y_poison_next).astype(int)
            pbar.set_postfix({'Optimizer': msg})

            if step > 1 and np.all(y_poison_next == y_poison):
                break
            y_poison = y_poison_next

            dataloader = self.numpy2dataloader(X_train, y_poison)
            train_model(model, dataloader, optimizer=optimizer, loss_fn=loss_fn, device=self.device, max_epochs=MAX_EPOCHS)
            alpha = get_dual_loss(model, X_train_tensor, self.device)

        return y_poison

    def apply_poisoning(self, file_path, advx_range):
        X, y, cols = open_csv(file_path)
        y = np.where(y == -1, 0, y)
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        n_features = X.shape[1]
        model = SimpleModel(n_features, hidden_dim=HIDDEN_LAYER, output_dim=2).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
        loss_fn = nn.CrossEntropyLoss()

        # Train initial baseline on the FULL dataset
        self.logger.info(f"Training clean baseline for {dataname}...")
        dataloader_full = self.numpy2dataloader(X, y)
        train_model(model, dataloader_full, optimizer, loss_fn, self.device, MAX_EPOCHS)

        path_poison_data_list = []

        for rate in advx_range:
            path_poison_data = f'{path_output_base}_falfa_nn_{rate:.2f}.csv'
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
            else:
                self.logger.info(f'     Generating {rate * 100:.0f}% poison data via FALFA NN...')
                y_flip = self.get_y_flip(model, X, y, rate, optimizer, loss_fn)
                to_csv(X, y_flip, cols, path_poison_data)

            path_poison_data_list.append(path_poison_data)

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
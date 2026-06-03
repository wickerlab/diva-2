import os
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim

from scripts.base_poisoner import BasePoisoner
from scripts.utils.utils import to_csv
from .utils.gradient_matching import witches_brew_optimize

class WitchesBrewPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="witches_brew", base_folder=base_folder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the ResNet18 Latent Extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        self.resnet.eval()
        self.latent_extractor = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

    def apply_poisoning(self, file_path, advx_range):
        pt_file_path = file_path.replace("clean_data", "raw_images").replace("_clean.csv", ".pt")
        data = torch.load(pt_file_path)
        X_images = data["X"]
        y_labels = data["y"]
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        idx_0 = torch.where(y_labels == 0)[0]
        idx_1 = torch.where(y_labels == 1)[0]
        
        metadata_list = []

        # =========================================================
        # NEW: SURROGATE MODEL PRE-TRAINING (The "Smart Head")
        # =========================================================
        self.logger.info(f"     Pre-training Surrogate Linear Head for {dataname}...")
        
        # 1. Extract latents for the clean dataset quickly
        clean_latents = []
        with torch.no_grad():
            for i in range(0, len(X_images), 128):
                batch = X_images[i:i+128].to(self.device)
                clean_latents.append(self.latent_extractor(batch).squeeze())
        clean_latents = torch.cat(clean_latents)
        
        # 2. Train a Linear Head to convergence (100 epochs)
        linear_head = nn.Linear(512, 2).to(self.device)
        optimizer = optim.Adam(linear_head.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        linear_head.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = linear_head(clean_latents)
            loss = criterion(outputs, y_labels.to(self.device))
            loss.backward()
            optimizer.step()
            
        linear_head.eval()
        
        # 3. Assemble the highly accurate Victim Surrogate Model
        victim_model = nn.Sequential(
            self.latent_extractor,
            nn.Flatten(),
            linear_head
        ).eval()
        # =========================================================

        for rate in advx_range:
            path_poison_data = f'{path_output_base}_witches_brew_{rate:.2f}.csv'
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
                metadata_list.append({"Data": dataname, "Path": path_poison_data, "Method": self.name, "Rate": rate, "Is_Poisoned": 1 if rate > 0 else 0})
                continue
                
            n_poison = int(len(X_images) * rate)

            if n_poison == 0:
                X_final_images = X_images
                y_final = y_labels
            else:
                self.logger.info(f'     Generating {rate * 100:.0f}% poison via Witches Brew...')
                
                # 1. Split budget & calculate targets using the Square Root rule
                n_poison_0 = n_poison // 2
                n_poison_1 = n_poison - n_poison_0
                
                n_targets_1 = max(1, int(np.ceil(np.sqrt(n_poison_0)))) # Class 0 attacks Class 1
                n_targets_0 = max(1, int(np.ceil(np.sqrt(n_poison_1)))) # Class 1 attacks Class 0
                
                # 2. Select base images to be corrupted
                base_idx_0 = idx_0[torch.randperm(len(idx_0))[:n_poison_0]]
                base_idx_1 = idx_1[torch.randperm(len(idx_1))[:n_poison_1]]
                
                # 3. Select target images
                target_idx_1 = idx_1[torch.randperm(len(idx_1))[:n_targets_1]]
                target_idx_0 = idx_0[torch.randperm(len(idx_0))[:n_targets_0]]
                
                # 4. Chunk the base indices so each chunk assigns to one target
                chunks_0 = torch.tensor_split(base_idx_0, n_targets_1)
                chunks_1 = torch.tensor_split(base_idx_1, n_targets_0)
                
                X_final_images = X_images.clone()

                # --- EXECUTE: Class 0 attacking Class 1 ---
                for i, chunk in enumerate(chunks_0):
                    if len(chunk) == 0: continue
                    b_images = X_images[chunk]
                    b_labels = y_labels[chunk]
                    t_image = X_images[target_idx_1[i]]
                    t_label = torch.tensor(0) 
                    
                    p_images = witches_brew_optimize(victim_model, b_images, b_labels, t_image, t_label)
                    X_final_images[chunk] = p_images.cpu()
                    
                # --- EXECUTE: Class 1 attacking Class 0 ---
                for i, chunk in enumerate(chunks_1):
                    if len(chunk) == 0: continue
                    b_images = X_images[chunk]
                    b_labels = y_labels[chunk]
                    t_image = X_images[target_idx_0[i]]
                    t_label = torch.tensor(1) 
                    
                    p_images = witches_brew_optimize(victim_model, b_images, b_labels, t_image, t_label)
                    X_final_images[chunk] = p_images.cpu()
                
                y_final = y_labels.clone()

            # --- LATENT CONVERSION ---
            self.logger.info(f"     Extracting Latent features for rate {rate:.2f}...")
            latent_vectors = []
            
            with torch.no_grad():
                for i in range(0, len(X_final_images), 128):
                    batch = X_final_images[i:i+128].to(self.device)
                    feats = self.latent_extractor(batch).squeeze()
                    latent_vectors.append(feats.cpu())
                    
            X_tabular = torch.cat(latent_vectors).numpy()
            y_tabular = y_final.numpy()
            
            cols = [f"feature_{i}" for i in range(X_tabular.shape[1])]
            to_csv(X_tabular, y_tabular, cols, path_poison_data)
            
            metadata_list.append({
                "Data": dataname, "Path": path_poison_data, "Method": self.name, "Rate": rate, "Is_Poisoned": 1 if rate > 0 else 0
            })

        return metadata_list
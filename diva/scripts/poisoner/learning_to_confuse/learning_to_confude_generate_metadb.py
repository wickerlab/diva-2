import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.base_poisoner import BasePoisoner
from scripts.utils.utils import to_csv

# ==========================================
# 1. The Generative Autoencoder Architecture
# ==========================================
class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()
        # Encoder: 32x32 -> 16x16 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Decoder: 8x8 -> 16x16 -> 32x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# 2. The Poisoner Class
# ==========================================
class AutoEncoderPoisoner(BasePoisoner):
    """
    Implements a Generative Poisoning Attack based on:
    Feng, J., Cai, Q. Z., & Zhou, Z. H. (2019). Learning to Confuse: 
    Generating Training Time Adversarial Data with Auto-Encoder.
    """
    def __init__(self, base_folder):
        super().__init__(name="learning_to_confuse", base_folder=base_folder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the ResNet18 Latent Extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        self.latent_extractor = torch.nn.Sequential(*(list(self.resnet.children())[:-1])).eval()

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

        # ---------------------------------------------------------
        # PRE-TRAINING: The Victim Surrogate Model
        # ---------------------------------------------------------
        self.logger.info(f"     Pre-training Surrogate Linear Head for {dataname}...")
        clean_latents = []
        with torch.no_grad():
            for i in range(0, len(X_images), 128):
                batch = X_images[i:i+128].to(self.device)
                clean_latents.append(self.latent_extractor(batch).squeeze())
        clean_latents = torch.cat(clean_latents)
        
        linear_head = nn.Linear(512, 2).to(self.device)
        optimizer_surrogate = optim.Adam(linear_head.parameters(), lr=0.01)
        criterion_ce = nn.CrossEntropyLoss()
        
        linear_head.train()
        for _ in range(80): # Train surrogate quickly
            optimizer_surrogate.zero_grad()
            outputs = linear_head(clean_latents)
            loss = criterion_ce(outputs, y_labels.to(self.device))
            loss.backward()
            optimizer_surrogate.step()
            
        victim_model = nn.Sequential(self.latent_extractor, nn.Flatten(), linear_head).eval()

        # ---------------------------------------------------------
        # POISON GENERATION LOOP
        # ---------------------------------------------------------
        for rate in advx_range:
            path_poison_data = f'{path_output_base}_learningtoconfuse_{rate:.2f}.csv'
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
                metadata_list.append({"Data": dataname, "Path": path_poison_data, "Method": self.name, "Rate": rate, "Is_Poisoned": 1 if rate > 0 else 0})
                continue
                
            n_poison = int(len(X_images) * rate)

            if n_poison == 0:
                X_final_images = X_images.clone()
                y_final = y_labels.clone()
            else:
                self.logger.info(f'     Generating {rate * 100:.0f}% poison via Generative AutoEncoder...')
                
                # Split budget for Symmetric Attack
                n_poison_0 = n_poison // 2
                n_poison_1 = n_poison - n_poison_0
                
                base_idx_0 = idx_0[torch.randperm(len(idx_0))[:n_poison_0]]
                base_idx_1 = idx_1[torch.randperm(len(idx_1))[:n_poison_1]]
                
                X_final_images = X_images.clone()
                
                # We use a balanced loss: 1.0 for visual similarity, 0.5 for adversarial confusion
                criterion_mse = nn.MSELoss()
                adv_weight = 0.5 

                # --- TRAIN & GENERATE: Class 0 -> Confuse as Class 1 ---
                if len(base_idx_0) > 0:
                    ae_0 = SimpleAE().to(self.device)
                    opt_ae_0 = optim.Adam(ae_0.parameters(), lr=0.005)
                    b_images_0 = X_images[base_idx_0].to(self.device)
                    target_labels_1 = torch.ones(len(b_images_0), dtype=torch.long, device=self.device) # Target Class 1
                    
                    ae_0.train()
                    for epoch in range(100): # 100 epochs is enough for a tiny CAE
                        opt_ae_0.zero_grad()
                        p_images = ae_0(b_images_0)
                        
                        loss_recon = criterion_mse(p_images, b_images_0)
                        loss_adv = criterion_ce(victim_model(p_images), target_labels_1)
                        
                        loss = loss_recon + (adv_weight * loss_adv)
                        loss.backward()
                        opt_ae_0.step()
                        
                    ae_0.eval()
                    with torch.no_grad():
                        X_final_images[base_idx_0] = ae_0(b_images_0).cpu()

                # --- TRAIN & GENERATE: Class 1 -> Confuse as Class 0 ---
                if len(base_idx_1) > 0:
                    ae_1 = SimpleAE().to(self.device)
                    opt_ae_1 = optim.Adam(ae_1.parameters(), lr=0.005)
                    b_images_1 = X_images[base_idx_1].to(self.device)
                    target_labels_0 = torch.zeros(len(b_images_1), dtype=torch.long, device=self.device) # Target Class 0
                    
                    ae_1.train()
                    for epoch in range(100):
                        opt_ae_1.zero_grad()
                        p_images = ae_1(b_images_1)
                        
                        loss_recon = criterion_mse(p_images, b_images_1)
                        loss_adv = criterion_ce(victim_model(p_images), target_labels_0)
                        
                        loss = loss_recon + (adv_weight * loss_adv)
                        loss.backward()
                        opt_ae_1.step()
                        
                    ae_1.eval()
                    with torch.no_grad():
                        X_final_images[base_idx_1] = ae_1(b_images_1).cpu()
                
                y_final = y_labels.clone()

            # ---------------------------------------------------------
            # LATENT CONVERSION & SAVING
            # ---------------------------------------------------------
            self.logger.info(f"     Extracting Latent features for Generative rate {rate:.2f}...")
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
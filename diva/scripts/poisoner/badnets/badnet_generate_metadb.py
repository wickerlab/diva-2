import os
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.base_poisoner import BasePoisoner
from scripts.utils.utils import to_csv

class BadNetsPoisoner(BasePoisoner):
    """
    Implements a Symmetric Dirty-Label Backdoor Attack based on:
    Gu, T., Dolan-Gavitt, B., & Garg, S. (2017). BadNets: Identifying 
    Vulnerabilities in the Machine Learning Model Supply Chain.
    """
    def __init__(self, base_folder):
        super().__init__(name="badnets", base_folder=base_folder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the ResNet18 Latent Extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        self.latent_extractor = torch.nn.Sequential(*(list(self.resnet.children())[:-1])).eval()

    def apply_poisoning(self, file_path, advx_range):
        # Swap the clean.csv path back to the raw .pt PyTorch tensors
        pt_file_path = file_path.replace("clean_data", "raw_images").replace("_clean.csv", ".pt")
        data = torch.load(pt_file_path)
        
        X_images = data["X"]
        y_labels = data["y"]
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        idx_0 = torch.where(y_labels == 0)[0]
        idx_1 = torch.where(y_labels == 1)[0]
        
        metadata_list = []

        for rate in advx_range:
            path_poison_data = f'{path_output_base}_badnets_{rate:.2f}.csv'
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
                metadata_list.append({"Data": dataname, "Path": path_poison_data, "Method": self.name, "Rate": rate, "Is_Poisoned": 1 if rate > 0 else 0})
                continue
                
            n_poison = int(len(X_images) * rate)

            if n_poison == 0:
                X_final_images = X_images.clone()
                y_final = y_labels.clone()
            else:
                self.logger.info(f'     Generating {rate * 100:.0f}% poison via Randomized Symmetric BadNets...')
                
                # 1. Split the budget to maintain perfect class balance
                n_poison_0 = n_poison // 2
                n_poison_1 = n_poison - n_poison_0
                
                # 2. Select random indices to corrupt
                base_idx_0 = idx_0[torch.randperm(len(idx_0))[:n_poison_0]]
                base_idx_1 = idx_1[torch.randperm(len(idx_1))[:n_poison_1]]
                
                X_final_images = X_images.clone()
                y_final = y_labels.clone()
                
                # 3. Dynamically Generate Dataset-Specific Triggers
                # We randomize the trigger PER DATASET to force the meta-learner to generalize
                _, _, H, W = X_images.shape
                
                # Randomize Size (between 3x3 and 6x6)
                t_size_0 = np.random.randint(3, 7)
                t_size_1 = np.random.randint(3, 7)
                
                # Randomize Intensity (Bright or Dark artifacts)
                intensity_0 = np.random.uniform(1.5, 3.0) * np.random.choice([1, -1])
                intensity_1 = np.random.uniform(1.5, 3.0) * np.random.choice([1, -1])
                
                # Randomize Placement (ensuring it fits within the image boundaries)
                h_0, w_0 = np.random.randint(0, H - t_size_0), np.random.randint(0, W - t_size_0)
                h_1, w_1 = np.random.randint(0, H - t_size_1), np.random.randint(0, W - t_size_1)
                
                # 4. Apply Class 0 -> Class 1 Attack
                if len(base_idx_0) > 0:
                    # Apply randomized trigger across all channels
                    X_final_images[base_idx_0, :, h_0:h_0+t_size_0, w_0:w_0+t_size_0] = intensity_0
                    y_final[base_idx_0] = 1 # Dirty-Label flip
                
                # 5. Apply Class 1 -> Class 0 Attack
                if len(base_idx_1) > 0:
                    # Apply randomized trigger across all channels
                    X_final_images[base_idx_1, :, h_1:h_1+t_size_1, w_1:w_1+t_size_1] = intensity_1
                    y_final[base_idx_1] = 0 # Dirty-Label flip

            # --- LATENT CONVERSION ---
            self.logger.info(f"     Extracting Latent features for BadNets rate {rate:.2f}...")
            latent_vectors = []
            
            with torch.no_grad():
                # Process in batches to save VRAM
                for i in range(0, len(X_final_images), 128):
                    batch = X_final_images[i:i+128].to(self.device)
                    feats = self.latent_extractor(batch).squeeze()
                    latent_vectors.append(feats.cpu())
                    
            X_tabular = torch.cat(latent_vectors).numpy()
            y_tabular = y_final.numpy()
            
            # Format as Tabular Data
            cols = [f"feature_{i}" for i in range(X_tabular.shape[1])]
            to_csv(X_tabular, y_tabular, cols, path_poison_data)
            
            metadata_list.append({
                "Data": dataname, "Path": path_poison_data, "Method": self.name, "Rate": rate, "Is_Poisoned": 1 if rate > 0 else 0
            })

        return metadata_list
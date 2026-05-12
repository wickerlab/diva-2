import os
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.base_poisoner import BasePoisoner
from scripts.utils.utils import to_csv
from .utils.polytope_optimization import bullseye_polytope_optimize

class BullseyePolytopePoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="bullseye_polytope", base_folder=base_folder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the Latent Extractor
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
            path_poison_data = f'{path_output_base}_bullseye_polytope_{rate:.2f}.csv'
            
            if os.path.exists(path_poison_data):
                self.logger.info(f'     Rate {rate:.2f}: Already generated. Skipping.')
                metadata_list.append({"Data": dataname, "Path": path_poison_data, "Method": self.name, "Rate": rate, "Is_Poisoned": 1 if rate > 0 else 0})
                continue
                
            n_poison = int(len(X_images) * rate)

            if n_poison == 0:
                X_final_images = X_images
                y_final = y_labels
            else:
                self.logger.info(f'     Generating {rate * 100:.0f}% poison via Bullseye Polytope...')
                
                # Target an image from class 1
                target_idx = idx_1[0] 
                target_image = X_images[target_idx].to(self.device)
                
                # Grab base images from class 0 to corrupt
                base_indices = idx_0[torch.randperm(len(idx_0))[:n_poison]]
                base_images = X_images[base_indices].to(self.device)
                base_labels = y_labels[base_indices]
                
                # Batch execution (creating "sub-polytopes" of 128 to save VRAM)
                poisoned_batches = []
                batch_size = 128 
                
                for i in range(0, len(base_images), batch_size):
                    batch_base = base_images[i:i+batch_size]
                    batch_poisoned = bullseye_polytope_optimize(
                        self.latent_extractor, batch_base, target_image
                    )
                    poisoned_batches.append(batch_poisoned.cpu())
                    
                    del batch_base, batch_poisoned
                    torch.cuda.empty_cache()
                    
                poisoned_images = torch.cat(poisoned_batches)
                
                # Inject poisons back into the dataset with the WRONG (clean) label
                X_final_images = torch.cat([X_images, poisoned_images])
                y_final = torch.cat([y_labels, base_labels.cpu()]) 

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
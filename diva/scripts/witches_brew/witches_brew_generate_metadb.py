import os
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from pathlib import Path
import concurrent.futures

from scripts.base_poisoner import BasePoisoner
from scripts.utils.utils import to_csv
from .utils.gradient_matching import witches_brew_optimize

class WitchesBrewPoisoner(BasePoisoner):
    def __init__(self, base_folder):
        super().__init__(name="witches_brew", base_folder=base_folder)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load a pretrained ResNet18 to act as the Victim Model and Latent Extractor
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        self.resnet.eval()
        
        # Strip the classification head to get the 512-d Latent Extractor
        self.latent_extractor = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        
        # Create a simple linear head for the Witches Brew victim model
        self.victim_model = torch.nn.Sequential(
            self.latent_extractor,
            torch.nn.Flatten(),
            torch.nn.Linear(512, 2)
        ).to(self.device)

    def apply_poisoning(self, file_path, advx_range):
        """
        Takes a .pt file containing raw images, attacks the pixel space, 
        extracts latent features, and saves them as CSVs for the DIVA pipeline.
        """
        # 1. Load the raw binary image tensors
        pt_file_path = file_path.replace("clean_data", "raw_images").replace("_clean.csv", ".pt")
        data = torch.load(pt_file_path)
        X_images = data["X"]
        y_labels = data["y"]
        
        dataname = Path(file_path).stem
        path_output_base = os.path.join(self.poisoned_dir, dataname)

        # Separate classes
        idx_0 = torch.where(y_labels == 0)[0]
        idx_1 = torch.where(y_labels == 1)[0]
        
        metadata_list = []

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
                self.logger.info(f'     Generating {rate * 100:.0f}% poison via Witches Brew Gradient Matching...')
                
                # Attacker tries to make a Class 1 target look like Class 0
                target_idx = idx_1[0]
                target_image = X_images[target_idx]
                target_label = torch.tensor(0) # Adversarial target label
                
                base_indices = idx_0[torch.randperm(len(idx_0))[:n_poison]]
                base_images = X_images[base_indices]
                base_labels = y_labels[base_indices]
                
                # Execute the complex mathematical image attack
                poisoned_images = witches_brew_optimize(
                    self.victim_model, base_images, base_labels, target_image, target_label
                )
                
                # Inject poisons back into the dataset
                X_final_images = torch.cat([X_images, poisoned_images.cpu()])
                y_final = torch.cat([y_labels, base_labels.cpu()]) # Clean label assigned to poison!

            # --- LATENT CONVERSION FOR DIVA TABULAR PIPELINE ---
            self.logger.info(f"     Extracting Latent features for rate {rate:.2f}...")
            latent_vectors = []
            
            with torch.no_grad():
                # Process in batches to avoid VRAM overflow
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
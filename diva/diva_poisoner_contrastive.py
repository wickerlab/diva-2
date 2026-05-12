import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Diva_Attack_Eval")

# =============================================================================
# Model & Dataset Definitions 
# =============================================================================
class DeepSetMetaExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepSetMetaExtractor, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        phi_out = self.phi(x)               
        pooled_rep = torch.mean(phi_out, dim=1) 
        meta_features = self.rho(pooled_rep)    
        return meta_features

class SurrogatePoisonDetector(nn.Module):
    def __init__(self, meta_feature_dim):
        super(SurrogatePoisonDetector, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(meta_feature_dim), # Automatically standardizes the meta-features
            nn.Linear(meta_feature_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1) # Outputs raw logits
        )

    def forward(self, meta_features):
        return self.network(meta_features)

class PoisonMetaDataset(Dataset):
    def __init__(self, csv_file, num_samples, input_dim, meta_cols):
        self.meta_df = pd.read_csv(csv_file)
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.meta_cols = meta_cols

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        dataset_path = row['Path']
        is_poisoned = float(row['Is_Poisoned'])
        
        meta_vals = row[self.meta_cols].values.astype(float)
        meta_vals = np.nan_to_num(meta_vals, nan=0.0, posinf=0.0, neginf=0.0)
        target_meta = torch.tensor(meta_vals, dtype=torch.float32)
        
        try:
            raw_df = pd.read_csv(dataset_path).apply(pd.to_numeric, errors='coerce')
            raw_data = raw_df.values
            raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            scaler = StandardScaler()
            raw_data = scaler.fit_transform(raw_data)
            raw_tensor = torch.tensor(raw_data, dtype=torch.float32)
            
            if raw_tensor.shape[0] > self.num_samples:
                indices = torch.randperm(raw_tensor.shape[0])[:self.num_samples]
                raw_tensor = raw_tensor[indices]
            elif raw_tensor.shape[0] < self.num_samples:
                padding = torch.zeros((self.num_samples - raw_tensor.shape[0], raw_tensor.shape[1]))
                raw_tensor = torch.cat([raw_tensor, padding], dim=0)
                
            if raw_tensor.shape[1] > self.input_dim:
                raw_tensor = raw_tensor[:, :self.input_dim]
            elif raw_tensor.shape[1] < self.input_dim:
                padding = torch.zeros((self.num_samples, self.input_dim - raw_tensor.shape[1]))
                raw_tensor = torch.cat([raw_tensor, padding], dim=1)
                
        except Exception:
            raw_tensor = torch.zeros((self.num_samples, self.input_dim))
            
        label_tensor = torch.tensor([is_poisoned], dtype=torch.float32)
        return raw_tensor, label_tensor, target_meta

class VictimModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

# Helper function to evaluate accuracy drops dynamically
def train_and_eval_victim(X_train, y_train, X_test, y_test, num_classes, device):
    model = VictimModel(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    return acc

# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_file = "./data/meta_db_universal.csv"
    
    mimic_save_path = "./data/mimic_model.pth"
    detector_save_path = "./data/surrogate_detector.pth"
    
    num_samples = 2000       
    num_features = 100       
    input_dim = num_features + 1 
    hidden_dim = 64
    batch_size = 16

    logger.info(f"--- Starting Load, Attack & Evaluate Pipeline on {device} ---")

    meta_cols = ['best_node.mean', 'best_node.sd', 'cls_coef', 'density', 'f1.mean'] 
    meta_feature_dim = len(meta_cols)
    
    mimic_model = DeepSetMetaExtractor(input_dim, hidden_dim, meta_feature_dim).to(device)
    detector = SurrogatePoisonDetector(meta_feature_dim).to(device)

    # -------------------------------------------------------------------------
    # 1. LOAD OR TRAIN MODELS
    # -------------------------------------------------------------------------
    if os.path.exists(mimic_save_path) and os.path.exists(detector_save_path):
        logger.info("Loading pre-trained Mimic Model and Surrogate Detector...")
        mimic_model.load_state_dict(torch.load(mimic_save_path, map_location=device, weights_only=True))
        detector.load_state_dict(torch.load(detector_save_path, map_location=device, weights_only=True))
    else:
        logger.info("Pre-trained models not found. Initiating full training pipeline...")
        dataset = PoisonMetaDataset(csv_file, num_samples, input_dim, meta_cols)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # PHASE 1: Pre-Train Mimic Model
        logger.info("\n[PHASE 1] Training DeepSet Meta-Feature Extractor...")
        mimic_optimizer = optim.Adam(mimic_model.parameters(), lr=1e-3)
        mimic_criterion = nn.MSELoss()
        
        mimic_model.train()
        for epoch in range(20): 
            epoch_loss = 0.0
            for b_data, _, b_target_meta in dataloader:
                b_data, b_target_meta = b_data.to(device), b_target_meta.to(device)
                
                mimic_optimizer.zero_grad()
                preds = mimic_model(b_data)
                loss = mimic_criterion(preds, b_target_meta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mimic_model.parameters(), max_norm=1.0)
                mimic_optimizer.step()
                epoch_loss += loss.item()
            logger.info(f"Phase 1 Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")

        # PHASE 2: Train Surrogate Detector (With RAM Cache)
        logger.info("\n[PHASE 2] Training Surrogate Poison Detector...")
        det_optimizer = optim.Adam(detector.parameters(), lr=1e-3)
        
        num_clean = len(dataset.meta_df[dataset.meta_df['Is_Poisoned'] == 0])
        num_poisoned = max(1, len(dataset.meta_df[dataset.meta_df['Is_Poisoned'] == 1]))
        pos_weight = torch.tensor([num_clean / num_poisoned]).to(device)
        det_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        mimic_model.eval()
        for param in mimic_model.parameters():
            param.requires_grad = False

        logger.info("Extracting meta-features to RAM for high-speed training...")
        all_meta_feats = []
        all_labels = []
        
        with torch.no_grad():
            for b_data, b_labels, _ in dataloader:
                b_data = b_data.to(device)
                all_meta_feats.append(mimic_model(b_data).cpu())
                all_labels.append(b_labels.cpu())
                
        # Combine into pure RAM tensors
        X_meta_train = torch.cat(all_meta_feats, dim=0).to(device)
        y_meta_train = torch.cat(all_labels, dim=0).to(device)
        
        fast_dataset = torch.utils.data.TensorDataset(X_meta_train, y_meta_train)
        fast_dataloader = DataLoader(fast_dataset, batch_size=batch_size, shuffle=True)

        detector.train()
        for epoch in range(15): 
            epoch_loss = 0.0
            for b_meta, b_labels in fast_dataloader:
                det_optimizer.zero_grad()
                loss = det_criterion(detector(b_meta), b_labels)
                loss.backward()
                det_optimizer.step()
                epoch_loss += loss.item()
            logger.info(f"Phase 2 Epoch {epoch+1} Loss: {epoch_loss/len(fast_dataloader):.4f}")

        # Save models
        torch.save(mimic_model.state_dict(), mimic_save_path)
        torch.save(detector.state_dict(), detector_save_path)
        logger.info("Models trained and successfully saved!")
    
    # Freeze for evaluation/attack
    mimic_model.eval()
    detector.eval()
    for p in mimic_model.parameters(): p.requires_grad = False
    for p in detector.parameters(): p.requires_grad = False

    # -------------------------------------------------------------------------
    # 2. SELECT DATASETS
    # -------------------------------------------------------------------------
    logger.info("\n[PHASE 3] Selecting Datasets for Attack...")
    df_meta = pd.read_csv(csv_file)
    clean_df = df_meta[df_meta['Is_Poisoned'] == 0]
    
    hf_df = clean_df[clean_df['Path'].str.contains('hf_', case=False, na=False)]
    openml_df = clean_df[clean_df['Path'].str.contains('openml', case=False, na=False)]
    synth_df = clean_df[clean_df['Path'].str.contains('synthetic', case=False, na=False)]
    
    n_datasets = 300

    n_hf, n_op, n_sy = min(n_datasets//3, len(hf_df)), min(n_datasets//3, len(openml_df)), min(n_datasets//3, len(synth_df))
    selected_df = pd.concat([
        hf_df.sample(n_hf, random_state=42), 
        openml_df.sample(n_op, random_state=42), 
        synth_df.sample(n_sy, random_state=42)
    ])
    
    if len(selected_df) < n_datasets:
        remaining = n_datasets - len(selected_df)
        pool = clean_df.drop(selected_df.index)
        if len(pool) > 0:
            selected_df = pd.concat([selected_df, pool.sample(min(remaining, len(pool)), random_state=42)])

    logger.info(f"Selected {len(selected_df)} datasets to attack.")
    
    out_dir = Path("data/poisoned_data/diva_attack2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tracking Results for Final Plots
    attack_results = []

    epochs_attack = 50
    epsilon = 2.0
    alpha_poison = 0.1
    lambda_evasion = 5.0
    inner_lr = 0.05

    # -------------------------------------------------------------------------
    # 3. ATTACK LOOP
    # -------------------------------------------------------------------------
    for idx, row in selected_df.iterrows():
        orig_path = row['Path']
        if not os.path.exists(orig_path):
            continue

        try:
            df_real = pd.read_csv(orig_path).apply(pd.to_numeric, errors='coerce').fillna(0.0)
            X_real = df_real.iloc[:, :-1].values
            y_real = df_real.iloc[:, -1].values.astype(int) 
            
            rate = np.random.uniform(0.05, 0.30)
            num_poison = max(1, int(rate * len(X_real)))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_real)

            indices = np.random.permutation(len(X_scaled))
            poison_idx = indices[:num_poison]
            clean_idx = indices[num_poison:]

            X_orig = torch.tensor(X_scaled[poison_idx], dtype=torch.float32).to(device)
            X_poison_active = X_orig.clone().requires_grad_(True)
            X_clean_frozen = torch.tensor(X_scaled[clean_idx], dtype=torch.float32).to(device)
            
            y_poison_active_t = torch.tensor(y_real[poison_idx], dtype=torch.long).to(device)
            y_clean_frozen_t = torch.tensor(y_real[clean_idx], dtype=torch.long).to(device)

            num_classes = len(np.unique(y_real))
            victim = VictimModel(input_dim=X_real.shape[1], num_classes=num_classes).to(device)
            
            opt_poison = optim.Adam([X_poison_active], lr=alpha_poison)
            opt_victim = optim.Adam(victim.parameters(), lr=0.01)

            # --- Differentiable Padding Helper ---
            def get_detector_prob(X_p, X_c, y_p, y_c):
                X_comb = torch.cat([X_p, X_c], dim=0)
                y_comb = torch.cat([y_p, y_c], dim=0)
                dataset_tensor = torch.cat([X_comb, y_comb.unsqueeze(1).float()], dim=1) 
                
                padded = dataset_tensor.clone()
                if padded.shape[0] > num_samples:
                    padded = padded[:num_samples]
                elif padded.shape[0] < num_samples:
                    pad_z = torch.zeros((num_samples - padded.shape[0], padded.shape[1])).to(device)
                    padded = torch.cat([padded, pad_z], dim=0)
                    
                if padded.shape[1] > input_dim:
                    padded = padded[:, :input_dim]
                elif padded.shape[1] < input_dim:
                    pad_z = torch.zeros((padded.shape[0], input_dim - padded.shape[1])).to(device)
                    padded = torch.cat([padded, pad_z], dim=1)

                meta_feats = mimic_model(padded.unsqueeze(0))
                return torch.sigmoid(detector(meta_feats).squeeze())

            # Baseline Clean Probability
            with torch.no_grad():
                prob_clean = get_detector_prob(X_orig, X_clean_frozen, y_poison_active_t, y_clean_frozen_t).item()

            # --- Optimization Loop ---
            for epoch in range(epochs_attack):
                X_combined = torch.cat([X_poison_active, X_clean_frozen], dim=0)
                y_combined = torch.cat([y_poison_active_t, y_clean_frozen_t], dim=0)

                victim.train()
                opt_victim.zero_grad()
                out_train = victim(X_combined.detach()) 
                loss_vic = F.cross_entropy(out_train, y_combined)
                loss_vic.backward()
                opt_victim.step()

                opt_poison.zero_grad()
                W, b = victim.linear.weight, victim.linear.bias
                logits_train = X_combined @ W.t() + b
                loss_train_unrolled = F.cross_entropy(logits_train, y_combined)
                
                grad_W, grad_b = torch.autograd.grad(loss_train_unrolled, [W, b], create_graph=True)
                W_fast = W - inner_lr * grad_W
                b_fast = b - inner_lr * grad_b
                
                logits_val = X_clean_frozen @ W_fast.t() + b_fast
                loss_avail = -F.cross_entropy(logits_val, y_clean_frozen_t)

                poison_prob = get_detector_prob(X_poison_active, X_clean_frozen, y_poison_active_t, y_clean_frozen_t)
                loss_evasion = poison_prob

                total_loss = loss_avail + (lambda_evasion * loss_evasion)
                total_loss.backward()
                opt_poison.step()

                with torch.no_grad():
                    perturbation = torch.clamp(X_poison_active - X_orig, min=-epsilon, max=epsilon)
                    X_poison_active.copy_(X_orig + perturbation)

            # --- Final Metric Collection ---
            X_poison_final = X_poison_active.detach()
            
            # Evaluate Accuracy Drop
            acc_clean = train_and_eval_victim(X_orig, y_poison_active_t, X_clean_frozen, y_clean_frozen_t, num_classes, device)
            acc_poison = train_and_eval_victim(X_poison_final, y_poison_active_t, X_clean_frozen, y_clean_frozen_t, num_classes, device)
            
            # Evaluate Final Poison Probability
            with torch.no_grad():
                prob_poison = get_detector_prob(X_poison_final, X_clean_frozen, y_poison_active_t, y_clean_frozen_t).item()
                perturb_norm = torch.norm(X_poison_final - X_orig, p=2, dim=1).mean().item()

            attack_results.append({
                'Dataset': os.path.basename(orig_path),
                'Rate': rate,
                'Prob_Clean': prob_clean,
                'Prob_Poison': prob_poison,
                'Prob_Diff': prob_poison - prob_clean,
                'Acc_Clean': acc_clean,
                'Acc_Poison': acc_poison,
                'Acc_Drop': acc_clean - acc_poison,
                'Avg_Perturbation': perturb_norm
            })

            # Save File
            X_poison_final_unscaled = scaler.inverse_transform(X_poison_final.cpu().numpy())
            df_real.iloc[poison_idx, :-1] = X_poison_final_unscaled
            df_real.to_csv(out_dir / f"poisoned_{os.path.basename(orig_path)}_diva_poisoner2_{rate:.2}.csv", index=False)
            
            logger.info(f"Done {os.path.basename(orig_path)} | Drop: {acc_clean - acc_poison:.2%} | Stealth Diff: {prob_poison - prob_clean:.4f}")
            
        except Exception as e:
            logger.warning(f"Skipped {orig_path}: {e}")

    # =========================================================================
    # 4. RESULTS VISUALIZATION (PLOTTING)
    # =========================================================================
    logger.info("\n[PHASE 4] Generating Evaluation Plots...")
    
    if len(attack_results) > 0:
        res_df = pd.DataFrame(attack_results)
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Diva Attack Evaluation Dashboard', fontsize=18, fontweight='bold')

        # 1. Detector Output Difference (Poisoned - Clean)
        sns.histplot(res_df['Prob_Diff'], bins=20, kde=True, ax=axes[0, 0], color='purple')
        axes[0, 0].set_title('Detector Stealth: Prob Difference (Poison - Clean)\n< 0 means attack made it look cleaner')
        axes[0, 0].set_xlabel('Probability Shift')
        axes[0, 0].axvline(0, color='black', linestyle='--')

        # 2. Accuracy Degradation (Clean Acc - Poison Acc)
        sns.histplot(res_df['Acc_Drop'], bins=20, kde=True, ax=axes[0, 1], color='red')
        axes[0, 1].set_title('Availability Effectiveness: Accuracy Drop\n> 0 means successful degradation')
        axes[0, 1].set_xlabel('Accuracy Degradation (Absolute)')
        axes[0, 1].axvline(0, color='black', linestyle='--')

        # 3. Detector Output on Clean Dataset (Baseline Check)
        sns.histplot(res_df['Prob_Clean'], bins=20, kde=True, ax=axes[0, 2], color='blue')
        axes[0, 2].set_title('Baseline Check: Detector Prob on Clean Data\nShould be close to 0')
        axes[0, 2].set_xlabel('Clean Dataset Probability')

        # 4. Clean vs Poisoned Probabilities (Scatter)
        sns.scatterplot(x='Prob_Clean', y='Prob_Poison', data=res_df, ax=axes[1, 0], hue='Rate', palette='viridis')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1) # Diagonal line
        axes[1, 0].set_title('Stealth Correlation: Clean vs. Poison Prob')
        axes[1, 0].set_xlim(-0.05, 1.05)
        axes[1, 0].set_ylim(-0.05, 1.05)

        # 5. Poisoning Rate vs Accuracy Degradation
        sns.regplot(x='Rate', y='Acc_Drop', data=res_df, ax=axes[1, 1], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        axes[1, 1].set_title('Budget vs. Impact: Rate vs. Accuracy Drop')
        axes[1, 1].set_xlabel('Poisoning Rate (Budget)')
        axes[1, 1].set_ylabel('Accuracy Degradation')

        # 6. Perturbation Magnitude vs Detector Difference
        sns.scatterplot(x='Avg_Perturbation', y='Prob_Diff', data=res_df, ax=axes[1, 2], alpha=0.7, color='teal')
        axes[1, 2].set_title('Cost of Stealth: Perturbation L2 Norm vs. Prob Shift')
        axes[1, 2].set_xlabel('Average Perturbation (L2 Norm)')
        axes[1, 2].set_ylabel('Detector Probability Shift')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = "data/poisoned_data/diva_attack_evaluation.png"
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Plots successfully saved to: {plot_path}")
    else:
        logger.error("No valid attacks completed to generate plots.")

    logger.info("Pipeline Execution Complete!")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import random
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb # Added XGBoost

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Diva_Attack_Eval_XGB")

# =============================================================================
# Model Definitions
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
            nn.BatchNorm1d(meta_feature_dim),
            nn.Linear(meta_feature_dim, 64), nn.ReLU(), # Slightly wider to mimic trees better
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1) # Outputs raw logits
        )

    def forward(self, meta_features):
        return self.network(meta_features)

class PoisonMetaDataset(Dataset):
    # Added xgb_probs to provide soft labels for distillation
    def __init__(self, df, num_samples, input_dim, meta_cols, xgb_probs):
        self.meta_df = df.copy()
        self.meta_df['xgb_prob'] = xgb_probs
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.meta_cols = meta_cols

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        dataset_path = row['Path']
        
        # We now target the XGBoost probability (distillation) instead of the hard label
        xgb_prob = float(row['xgb_prob'])
        
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
            
        xgb_label_tensor = torch.tensor([xgb_prob], dtype=torch.float32)
        return raw_tensor, xgb_label_tensor, target_meta

class VictimModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

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
    detector_save_path = "./data/surrogate_detector_xgb_distilled.pth"
    
    num_samples = 2000       
    num_features = 100       
    input_dim = num_features + 1 
    hidden_dim = 64
    batch_size = 16

    logger.info(f"--- Starting Load, Attack & Evaluate Pipeline on {device} ---")

    meta_cols = ['best_node.mean', 'best_node.sd', 'cls_coef', 'density', 'f1.mean'] 
    meta_feature_dim = len(meta_cols)
    
    # -------------------------------------------------------------------------
    # 0. TRAIN THE TRUE XGBOOST DETECTOR
    # -------------------------------------------------------------------------
    logger.info("Training the True Non-Differentiable XGBoost Detector...")
    df_meta_full = pd.read_csv(csv_file)
    X_meta = df_meta_full[meta_cols].values
    y_meta = df_meta_full['Is_Poisoned'].values

    # Train actual XGBoost
    xgb_detector = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_detector.fit(X_meta, y_meta)
    
    # Get the "Soft Labels" (Probabilities) to train our PyTorch Surrogate
    xgb_soft_labels = xgb_detector.predict_proba(X_meta)[:, 1]
    
    # -------------------------------------------------------------------------
    # 1. LOAD OR TRAIN SURROGATE MODELS (DISTILLATION)
    # -------------------------------------------------------------------------
    mimic_model = DeepSetMetaExtractor(input_dim, hidden_dim, meta_feature_dim).to(device)
    detector = SurrogatePoisonDetector(meta_feature_dim).to(device)

    if os.path.exists(mimic_save_path) :
        logger.info("Loading pre-trained Distilled Models...")
        mimic_model.load_state_dict(torch.load(mimic_save_path, map_location=device, weights_only=True))
    else:
        logger.info("Distilling XGBoost into PyTorch Surrogate...")
        dataset = PoisonMetaDataset(df_meta_full, num_samples, input_dim, meta_cols, xgb_soft_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # PHASE 1: Pre-Train Mimic Model (Feature Extractor)
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
        torch.save(mimic_model.state_dict(), mimic_save_path)

    if os.path.exists(detector_save_path):
        detector.load_state_dict(torch.load(detector_save_path, map_location=device, weights_only=True))
    else:
        dataset = PoisonMetaDataset(df_meta_full, num_samples, input_dim, meta_cols, xgb_soft_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # PHASE 2: Train Surrogate Detector (Distillation from XGBoost)
        logger.info("\n[PHASE 2] Distilling XGBoost Probabilities into NN Surrogate...")
        det_optimizer = optim.Adam(detector.parameters(), lr=1e-3)
        
        # We use BCEWithLogitsLoss because our target (xgb_soft_labels) is a probability [0,1]
        det_criterion = nn.BCEWithLogitsLoss()

        mimic_model.eval()
        for param in mimic_model.parameters():
            param.requires_grad = False

        all_meta_feats, all_xgb_probs = [], []
        
        with torch.no_grad():
            for b_data, b_xgb_probs, _ in dataloader:
                b_data = b_data.to(device)
                all_meta_feats.append(mimic_model(b_data).cpu())
                all_xgb_probs.append(b_xgb_probs.cpu())
                
        X_meta_train = torch.cat(all_meta_feats, dim=0).to(device)
        y_xgb_train = torch.cat(all_xgb_probs, dim=0).to(device)
        
        fast_dataset = torch.utils.data.TensorDataset(X_meta_train, y_xgb_train)
        fast_dataloader = DataLoader(fast_dataset, batch_size=batch_size, shuffle=True)

        detector.train()
        for epoch in range(20): # Increased epochs slightly for better distillation
            epoch_loss = 0.0
            for b_meta, b_xgb_target in fast_dataloader:
                det_optimizer.zero_grad()
                loss = det_criterion(detector(b_meta), b_xgb_target)
                loss.backward()
                det_optimizer.step()
                epoch_loss += loss.item()
            logger.info(f"Phase 2 Epoch {epoch+1} Distillation Loss: {epoch_loss/len(fast_dataloader):.4f}")

        torch.save(detector.state_dict(), detector_save_path)
        logger.info("Models distilled and successfully saved!")
    
    # Freeze for evaluation/attack
    mimic_model.eval()
    detector.eval()
    for p in mimic_model.parameters(): p.requires_grad = False
    for p in detector.parameters(): p.requires_grad = False

    # -------------------------------------------------------------------------
    # 2.5. BENCHMARK SURROGATE FIDELITY
    # -------------------------------------------------------------------------
    logger.info("\n[BENCHMARK] Evaluating Surrogate Fidelity against True XGBoost...")
    
    # Sample up to 200 random datasets for a quick benchmark
    bench_df = df_meta_full.sample(n=min(200, len(df_meta_full)), random_state=99)
    bench_xgb_probs = xgb_detector.predict_proba(bench_df[meta_cols].values)[:, 1]
    
    bench_dataset = PoisonMetaDataset(bench_df, num_samples, input_dim, meta_cols, bench_xgb_probs)
    bench_loader = DataLoader(bench_dataset, batch_size=batch_size, shuffle=False)

    surrogate_probs = []
    true_xgb_probs = []

    with torch.no_grad():
        for b_data, b_xgb_probs, _ in bench_loader:
            b_data = b_data.to(device)
            # 1. Extract meta features using the frozen mimic model
            meta_feats = mimic_model(b_data)
            # 2. Get PyTorch Surrogate probability (using sigmoid to map logits to [0,1])
            preds = torch.sigmoid(detector(meta_feats).squeeze())
            
            # Catch edge case where batch size is 1 and squeeze removes the dimension
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            
            surrogate_probs.extend(preds.cpu().numpy().tolist())
            true_xgb_probs.extend(b_xgb_probs.squeeze().cpu().numpy().tolist())
            
    surrogate_probs = np.array(surrogate_probs)
    true_xgb_probs = np.array(true_xgb_probs)

    mae = np.mean(np.abs(surrogate_probs - true_xgb_probs))
    mse = np.mean((surrogate_probs - true_xgb_probs)**2)

    logger.info(f"Surrogate Benchmark on {len(bench_df)} datasets:")
    logger.info(f" -> Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f" -> Mean Squared Error (MSE):  {mse:.4f}")
    
    if mae < 0.05:
        logger.info(" -> Fidelity Check: PASSED. Surrogate highly mimics XGBoost.")
    elif mae < 0.15:
        logger.info(" -> Fidelity Check: OKAY. Surrogate roughly mimics XGBoost, attack might be noisy.")
    else:
        logger.warning(" -> Fidelity Check: FAILED. Distillation is poor. Consider training Phase 2 longer.")

    # -------------------------------------------------------------------------
    # 3. SELECT DATASETS (Originally Phase 3)
    # -------------------------------------------------------------------------
    logger.info("\n[PHASE 3] Selecting Datasets for Attack...")
    clean_df = df_meta_full[df_meta_full['Is_Poisoned'] == 0]
    
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
    
    out_dir = Path("data/poisoned_data/diva_attack_xgb")
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_results = []
    epochs_attack = 50
    epsilon = 2.0
    alpha_poison = 0.1
    lambda_evasion = 5.0
    inner_lr = 0.05

    # Helper to format data for meta extractor
    def get_padded_tensor(X_p, X_c, y_p, y_c):
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
        return padded

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

            def get_surrogate_prob(X_p, X_c, y_p, y_c):
                padded = get_padded_tensor(X_p, X_c, y_p, y_c)
                meta_feats = mimic_model(padded.unsqueeze(0))
                return torch.sigmoid(detector(meta_feats).squeeze())

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

                loss_evasion = get_surrogate_prob(X_poison_active, X_clean_frozen, y_poison_active_t, y_clean_frozen_t)

                total_loss = loss_avail + (lambda_evasion * loss_evasion)
                total_loss.backward()
                opt_poison.step()

                with torch.no_grad():
                    perturbation = torch.clamp(X_poison_active - X_orig, min=-epsilon, max=epsilon)
                    X_poison_active.copy_(X_orig + perturbation)

            # --- Final Metric Collection vs TRUE XGBOOST ---
            X_poison_final = X_poison_active.detach()
            
            # Evaluate Accuracy Drop
            acc_clean = train_and_eval_victim(X_orig, y_poison_active_t, X_clean_frozen, y_clean_frozen_t, num_classes, device)
            acc_poison = train_and_eval_victim(X_poison_final, y_poison_active_t, X_clean_frozen, y_clean_frozen_t, num_classes, device)
            
            with torch.no_grad():
                # 1. Evaluate baseline clean data on True XGBoost
                pad_clean = get_padded_tensor(X_orig, X_clean_frozen, y_poison_active_t, y_clean_frozen_t)
                meta_clean = mimic_model(pad_clean.unsqueeze(0)).cpu().numpy()
                prob_clean_xgb = xgb_detector.predict_proba(meta_clean)[0, 1]

                # 2. Evaluate poisoned data on True XGBoost
                pad_poison = get_padded_tensor(X_poison_final, X_clean_frozen, y_poison_active_t, y_clean_frozen_t)
                meta_poison = mimic_model(pad_poison.unsqueeze(0)).cpu().numpy()
                prob_poison_xgb = xgb_detector.predict_proba(meta_poison)[0, 1]

                perturb_norm = torch.norm(X_poison_final - X_orig, p=2, dim=1).mean().item()

            attack_results.append({
                'Dataset': os.path.basename(orig_path),
                'Rate': rate,
                'XGB_Prob_Clean': prob_clean_xgb,
                'XGB_Prob_Poison': prob_poison_xgb,
                'XGB_Prob_Diff': prob_poison_xgb - prob_clean_xgb,
                'Acc_Clean': acc_clean,
                'Acc_Poison': acc_poison,
                'Acc_Drop': acc_clean - acc_poison,
                'Avg_Perturbation': perturb_norm
            })

            # Save File
            X_poison_final_unscaled = scaler.inverse_transform(X_poison_final.cpu().numpy())
            df_real.iloc[poison_idx, :-1] = X_poison_final_unscaled
            df_real.to_csv(out_dir / f"poisoned_{os.path.basename(orig_path)}_xgb_evaded_{rate:.2}.csv", index=False)
            
            logger.info(f"Done {os.path.basename(orig_path)} | Drop: {acc_clean - acc_poison:.2%} | True XGB Prob: {prob_poison_xgb:.4f}")
            
        except Exception as e:
            logger.warning(f"Skipped {orig_path}: {e}")

    # =========================================================================
    # 4. RESULTS VISUALIZATION
    # =========================================================================
    logger.info("\n[PHASE 4] Generating Evaluation Plots...")
    
    if len(attack_results) > 0:
        res_df = pd.DataFrame(attack_results)
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('XGBoost Distillation Attack Evaluation Dashboard', fontsize=18, fontweight='bold')

        sns.histplot(res_df['XGB_Prob_Diff'], bins=20, kde=True, ax=axes[0, 0], color='purple')
        axes[0, 0].set_title('True XGBoost Stealth (Poison - Clean)\n< 0 means attack made it look cleaner')
        axes[0, 0].axvline(0, color='black', linestyle='--')

        sns.histplot(res_df['Acc_Drop'], bins=20, kde=True, ax=axes[0, 1], color='red')
        axes[0, 1].set_title('Victim Accuracy Drop\n> 0 means successful degradation')
        axes[0, 1].axvline(0, color='black', linestyle='--')

        sns.histplot(res_df['XGB_Prob_Clean'], bins=20, kde=True, ax=axes[0, 2], color='blue')
        axes[0, 2].set_title('True XGBoost Prob on Clean Data')

        sns.scatterplot(x='XGB_Prob_Clean', y='XGB_Prob_Poison', data=res_df, ax=axes[1, 0], hue='Rate', palette='viridis')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1) 
        axes[1, 0].set_title('Clean vs. Poison Prob on True XGBoost')
        axes[1, 0].set_xlim(-0.05, 1.05)
        axes[1, 0].set_ylim(-0.05, 1.05)

        sns.regplot(x='Rate', y='Acc_Drop', data=res_df, ax=axes[1, 1], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        axes[1, 1].set_title('Rate vs. Accuracy Drop')

        sns.scatterplot(x='Avg_Perturbation', y='XGB_Prob_Diff', data=res_df, ax=axes[1, 2], alpha=0.7, color='teal')
        axes[1, 2].set_title('Perturbation L2 Norm vs. XGBoost Prob Shift')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = "data/poisoned_data/xgb_attack_evaluation.png"
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Plots successfully saved to: {plot_path}")
    else:
        logger.error("No valid attacks completed to generate plots.")

    logger.info("Pipeline Execution Complete!")
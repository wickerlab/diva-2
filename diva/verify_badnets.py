import os
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

def verify_badnets_attack(clean_pt_path="data/raw_images/cifar10_0_vs_1.pt"):
    if not os.path.exists(clean_pt_path):
        print(f"Error: Could not find {clean_pt_path}. Please provide a valid .pt file path.")
        return

    print(f"Loading clean dataset from {clean_pt_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ResNet18 Latent Extractor
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    resnet.eval()
    latent_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1]))

    # Load Data
    data = torch.load(clean_pt_path)
    X_all = data["X"]
    y_all = data["y"]
    
    # Create a clean train/test split
    indices = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y_all)
    
    X_train_clean, y_train_clean = X_all[train_idx], y_all[train_idx]
    X_test_clean, y_test_clean = X_all[test_idx], y_all[test_idx]

    idx_0_train = torch.where(y_train_clean == 0)[0]
    idx_1_train = torch.where(y_train_clean == 1)[0]
    
    idx_0_test = torch.where(y_test_clean == 0)[0]
    idx_1_test = torch.where(y_test_clean == 1)[0]

    rates = [0.01, 0.03, 0.05, 0.08, 0.10]
    results = []

    # =========================================================
    # Evaluation Loop
    # =========================================================
    for rate in tqdm(rates, desc="Evaluating BadNets Rates"):
        n_poison = int(len(X_train_clean) * rate)
        if n_poison == 0: continue
        
        # 1. Budget Split (Symmetric)
        n_poison_0 = n_poison // 2
        n_poison_1 = n_poison - n_poison_0
        
        # 2. Select Indices to Poison
        base_idx_0 = idx_0_train[torch.randperm(len(idx_0_train))[:n_poison_0]]
        base_idx_1 = idx_1_train[torch.randperm(len(idx_1_train))[:n_poison_1]]
        
        X_train_poisoned = X_train_clean.clone()
        y_train_poisoned = y_train_clean.clone()
        
        # 3. Generate Random Triggers for this specific dataset simulation
        _, _, H, W = X_train_clean.shape
        t_size_0, t_size_1 = np.random.randint(3, 7), np.random.randint(3, 7)
        intensity_0 = np.random.uniform(1.5, 3.0) * np.random.choice([1, -1])
        intensity_1 = np.random.uniform(1.5, 3.0) * np.random.choice([1, -1])
        h_0, w_0 = np.random.randint(0, H - t_size_0), np.random.randint(0, W - t_size_0)
        h_1, w_1 = np.random.randint(0, H - t_size_1), np.random.randint(0, W - t_size_1)
        
        # 4. Apply Poisoning to Train Data
        if len(base_idx_0) > 0:
            X_train_poisoned[base_idx_0, :, h_0:h_0+t_size_0, w_0:w_0+t_size_0] = intensity_0
            y_train_poisoned[base_idx_0] = 1 # Flip label to 1
            
        if len(base_idx_1) > 0:
            X_train_poisoned[base_idx_1, :, h_1:h_1+t_size_1, w_1:w_1+t_size_1] = intensity_1
            y_train_poisoned[base_idx_1] = 0 # Flip label to 0
            
        # 5. Extract Train Latents & Train Victim
        train_latents = []
        with torch.no_grad():
            for i in range(0, len(X_train_poisoned), 128):
                batch = X_train_poisoned[i:i+128].to(device)
                train_latents.append(latent_extractor(batch).squeeze().cpu())
        X_train_tab = torch.cat(train_latents).numpy()
        y_train_tab = y_train_poisoned.numpy()

        clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            clf.fit(X_train_tab, y_train_tab)

        # 6. EVALUATE CLEAN HOLD OUT (Did it ruin the global accuracy?)
        test_latents_clean = []
        with torch.no_grad():
            for i in range(0, len(X_test_clean), 128):
                batch = X_test_clean[i:i+128].to(device)
                test_latents_clean.append(latent_extractor(batch).squeeze().cpu())
        X_test_tab_clean = torch.cat(test_latents_clean).numpy()
        y_test_tab_clean = y_test_clean.numpy()
        
        pred_clean = clf.predict(X_test_tab_clean)
        clean_miss_0 = np.mean(pred_clean[idx_0_test] != y_test_tab_clean[idx_0_test]) * 100
        clean_miss_1 = np.mean(pred_clean[idx_1_test] != y_test_tab_clean[idx_1_test]) * 100

        # 7. EVALUATE BACKDOOR SUCCESS (Did it learn the secret triggers?)
        # Create backdoored versions of the test sets
        X_test_bd_0 = X_test_clean[idx_0_test].clone()
        X_test_bd_1 = X_test_clean[idx_1_test].clone()
        
        X_test_bd_0[:, :, h_0:h_0+t_size_0, w_0:w_0+t_size_0] = intensity_0
        X_test_bd_1[:, :, h_1:h_1+t_size_1, w_1:w_1+t_size_1] = intensity_1
        
        latents_bd_0 = []
        latents_bd_1 = []
        with torch.no_grad():
            # Process Trigger 0 on Class 0
            for i in range(0, len(X_test_bd_0), 128):
                batch = X_test_bd_0[i:i+128].to(device)
                latents_bd_0.append(latent_extractor(batch).squeeze().cpu())
            # Process Trigger 1 on Class 1
            for i in range(0, len(X_test_bd_1), 128):
                batch = X_test_bd_1[i:i+128].to(device)
                latents_bd_1.append(latent_extractor(batch).squeeze().cpu())
                
        X_tab_bd_0 = torch.cat(latents_bd_0).numpy()
        X_tab_bd_1 = torch.cat(latents_bd_1).numpy()
        
        # ASR is the % of Backdoored Class 0 images predicted as Class 1
        asr_0 = np.mean(clf.predict(X_tab_bd_0) == 1) * 100
        # ASR is the % of Backdoored Class 1 images predicted as Class 0
        asr_1 = np.mean(clf.predict(X_tab_bd_1) == 0) * 100
        
        results.append({
            'Method': 'BadNets', 'Rate': rate, 
            'Backdoor_ASR_0': asr_0, 'Backdoor_ASR_1': asr_1,
            'Clean_Miss_0': clean_miss_0, 'Clean_Miss_1': clean_miss_1
        })

    # =========================================================
    # Plotting
    # =========================================================
    results_df = pd.DataFrame(results)
    
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Backdoor Attack Success Rate
    sns.lineplot(data=results_df, x='Rate', y='Backdoor_ASR_0', marker='o', linewidth=3, label="Trigger 0 (Class 0->1)", ax=ax1)
    sns.lineplot(data=results_df, x='Rate', y='Backdoor_ASR_1', marker='s', linewidth=3, label="Trigger 1 (Class 1->0)", ax=ax1)
    ax1.set_title("BadNets Backdoor ASR\n(Secret Trigger Effectiveness)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Poisoning Rate", fontsize=12)
    ax1.set_ylabel("Misclassification Rate (%)", fontsize=12)
    ax1.set_ylim(-5, 105)
    
    # Plot 2: Clean Dataset Misclassification (Global Degradation)
    sns.lineplot(data=results_df, x='Rate', y='Clean_Miss_0', marker='o', linewidth=3, linestyle="--", label="Clean Class 0", ax=ax2)
    sns.lineplot(data=results_df, x='Rate', y='Clean_Miss_1', marker='s', linewidth=3, linestyle="--", label="Clean Class 1", ax=ax2)
    ax2.set_title("Global Hyperplane Shift\n(Clean Image Misclassification)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Poisoning Rate", fontsize=12)
    ax2.set_ylabel("Misclassification Rate (%)", fontsize=12)
    ax2.set_ylim(-5, 105) 
    
    plt.tight_layout()
    plt.savefig("badnets_verification.png", dpi=300, bbox_inches='tight')
    print("\n✅ Verification complete! Saved plot to badnets_verification.png")

if __name__ == "__main__":
    verify_badnets_attack()
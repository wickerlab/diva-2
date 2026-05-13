import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

# Import your core optimization functions
from scripts.witches_brew.utils.gradient_matching import witches_brew_optimize
from scripts.poisoner.poison_frogs.utils.feature_collision import poison_frogs_optimize

def verify_targeted_attacks(clean_pt_path="data/raw_images/cifar10_1_vs_9.pt"):
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
    X_images_clean = data["X"]
    y_labels_clean = data["y"]
    
    idx_0 = torch.where(y_labels_clean == 0)[0]
    idx_1 = torch.where(y_labels_clean == 1)[0]

    rates = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    methods = ["poison_frogs", "witches_brew"]
    
    results = []

    # =========================================================
    # Pre-train Surrogate Model for Witches Brew
    # =========================================================
    print("\nPre-training Surrogate Linear Head for Witches Brew...")
    clean_latents = []
    with torch.no_grad():
        for i in range(0, len(X_images_clean), 128):
            batch = X_images_clean[i:i+128].to(device)
            clean_latents.append(latent_extractor(batch).squeeze())
    clean_latents = torch.cat(clean_latents)
    
    linear_head = nn.Linear(512, 2).to(device)
    optimizer = optim.Adam(linear_head.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    linear_head.train()
    for _ in range(100):
        optimizer.zero_grad()
        outputs = linear_head(clean_latents)
        loss = criterion(outputs, y_labels_clean.to(device))
        loss.backward()
        optimizer.step()
        
    linear_head.eval()
    victim_model = nn.Sequential(latent_extractor, nn.Flatten(), linear_head).eval()

    # =========================================================
    # Evaluation Loop
    # =========================================================
    for method in methods:
        print(f"\nEvaluating {method.upper()}...")
        for rate in tqdm(rates, desc=f"Rates for {method}"):
            n_poison = int(len(X_images_clean) * rate)
            if n_poison == 0: continue
            
            # 1. Budget Split (Square Root Rule)
            n_poison_0 = n_poison // 2
            n_poison_1 = n_poison - n_poison_0
            print(n_poison_0)
            n_targets_1 = max(1, int(np.ceil(np.sqrt(n_poison_0)))) 
            n_targets_0 = max(1, int(np.ceil(np.sqrt(n_poison_1)))) 
            
            # 2. Select Indices
            base_idx_0 = idx_0[torch.randperm(len(idx_0))[:n_poison_0]]
            base_idx_1 = idx_1[torch.randperm(len(idx_1))[:n_poison_1]]
            
            # WE SAVE THESE TARGET INDICES FOR EVALUATION
            target_idx_1 = idx_1[torch.randperm(len(idx_1))[:n_targets_1]]
            target_idx_0 = idx_0[torch.randperm(len(idx_0))[:n_targets_0]]
            
            chunks_0 = torch.tensor_split(base_idx_0, n_targets_1)
            chunks_1 = torch.tensor_split(base_idx_1, n_targets_0)
            
            X_final_images = X_images_clean.clone()

            # 3. Apply Poisoning
            if method == "poison_frogs":
                for i, chunk in enumerate(chunks_0):
                    if len(chunk) == 0: continue
                    b_images = X_images_clean[chunk].to(device)
                    t_image = X_images_clean[target_idx_1[i]].to(device)
                    for j in range(0, len(b_images), 128):
                        batch_poisoned = poison_frogs_optimize(latent_extractor, b_images[j:j+128], t_image)
                        X_final_images[chunk[j:j+128]] = batch_poisoned.cpu()
                        
                for i, chunk in enumerate(chunks_1):
                    if len(chunk) == 0: continue
                    b_images = X_images_clean[chunk].to(device)
                    t_image = X_images_clean[target_idx_0[i]].to(device)
                    for j in range(0, len(b_images), 128):
                        batch_poisoned = poison_frogs_optimize(latent_extractor, b_images[j:j+128], t_image)
                        X_final_images[chunk[j:j+128]] = batch_poisoned.cpu()
                        
            elif method == "witches_brew":
                for i, chunk in enumerate(chunks_0):
                    if len(chunk) == 0: continue
                    p_images = witches_brew_optimize(
                        victim_model, X_images_clean[chunk], y_labels_clean[chunk], 
                        X_images_clean[target_idx_1[i]], torch.tensor(0)
                    )
                    X_final_images[chunk] = p_images.cpu()
                    
                for i, chunk in enumerate(chunks_1):
                    if len(chunk) == 0: continue
                    p_images = witches_brew_optimize(
                        victim_model, X_images_clean[chunk], y_labels_clean[chunk], 
                        X_images_clean[target_idx_0[i]], torch.tensor(1)
                    )
                    X_final_images[chunk] = p_images.cpu()

            # 4. Extract Latents for Training Victim Model
            latent_vectors = []
            with torch.no_grad():
                for i in range(0, len(X_final_images), 128):
                    batch = X_final_images[i:i+128].to(device)
                    latent_vectors.append(latent_extractor(batch).squeeze().cpu())
            X_train = torch.cat(latent_vectors).numpy()
            y_train = y_labels_clean.numpy()

            # 5. Train Unregularized Linear Victim
            clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                clf.fit(X_train, y_train)

            # 6. EVALUATE TARGETS DIRECTLY
            with torch.no_grad():
                # Extract latents of the actual clean targets we selected
                t1_latents = latent_extractor(X_images_clean[target_idx_1].to(device)).squeeze().cpu().numpy()
                t0_latents = latent_extractor(X_images_clean[target_idx_0].to(device)).squeeze().cpu().numpy()
                
                # Reshape if only 1 target
                if len(t1_latents.shape) == 1: t1_latents = t1_latents.reshape(1, -1)
                if len(t0_latents.shape) == 1: t0_latents = t0_latents.reshape(1, -1)

            # A success means the target was forced into the OPPOSITE class
            pred_t1 = clf.predict(t1_latents)
            pred_t0 = clf.predict(t0_latents)
            
            asr_1 = np.mean(pred_t1 == 0) * 100 # Class 1 targeted by Class 0
            asr_0 = np.mean(pred_t0 == 1) * 100 # Class 0 targeted by Class 1
            
            results.append({
                'Method': method, 'Rate': rate, 
                'Targeted_ASR_Class_1': asr_1, 'Targeted_ASR_Class_0': asr_0,
                'Total_Targets': len(target_idx_1) + len(target_idx_0)
            })

    # =========================================================
    # Plotting
    # =========================================================
    results_df = pd.DataFrame(results)
    
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.lineplot(data=results_df, x='Rate', y='Targeted_ASR_Class_1', hue='Method', marker='o', linewidth=3, ax=ax1)
    ax1.set_title("Targeted ASR: Specific Class 1 Images predicted as 0", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Poisoning Rate", fontsize=12)
    ax1.set_ylabel("Target Misclassification Rate (%)", fontsize=12)
    ax1.set_ylim(-5, 105)
    
    sns.lineplot(data=results_df, x='Rate', y='Targeted_ASR_Class_0', hue='Method', marker='s', linewidth=3, ax=ax2)
    ax2.set_title("Targeted ASR: Specific Class 0 Images predicted as 1", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Poisoning Rate", fontsize=12)
    ax2.set_ylabel("Target Misclassification Rate (%)", fontsize=12)
    ax2.set_ylim(-5, 105) 
    
    plt.tight_layout()
    plt.savefig("targeted_attack_verification.png", dpi=300, bbox_inches='tight')
    print("\n✅ Verification complete! Saved plot to targeted_attack_verification.png")

if __name__ == "__main__":
    # You might need to change this path to match an actual file in your directory
    verify_targeted_attacks()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

def evaluate_attack_success_rate_mlp(meta_db_path="data/meta_db_image.csv"):
    if not os.path.exists(meta_db_path):
        print(f"Error: Could not find {meta_db_path}. Please check the path.")
        return
        
    print(f"Loading metadata from {meta_db_path}...")
    meta_df = pd.read_csv(meta_db_path)
    
    results = []
    
    # Filter for our specific targeted attacks
    targeted_methods = ['witches_brew', 'poison_frogs', 'bullseye_polytope']
    attack_df = meta_df[meta_df['Method'].isin(targeted_methods)]
    
    if attack_df.empty:
        print("No targeted attack datasets found in the MetaDB.")
        return

    for _, row in tqdm(attack_df.iterrows(), total=len(attack_df), desc="Evaluating ASR with MLP"):
        csv_path = row['Path']
        method = row['Method']
        rate = row['Rate']
        dataset_name = row['Data']
        
        if not os.path.exists(csv_path):
            continue
            
        # 1. Load the poisoned training data
        train_data = pd.read_csv(csv_path)
        X_train = train_data.drop('y', axis=1).values
        y_train = train_data['y'].values
        
        # 2. Train the Victim Model (2-Layer Non-Linear MLP)
        # Using 256 and 128 neurons to model complex latent space geometries
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128), 
            activation='relu', 
            solver='adam', 
            max_iter=1000, 
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # 3. Locate the Target Image(s)
        clean_row = meta_df[(meta_df['Data'] == dataset_name) & (meta_df['Method'] == 'clean')]
        if clean_row.empty:
            continue
            
        clean_csv_path = clean_row.iloc[0]['Path']
        clean_data = pd.read_csv(clean_csv_path)
        
        X_clean = clean_data.drop('y', axis=1).values
        y_clean = clean_data['y'].values
        
        # Our poisoners always target the FIRST Class 1 image in the clean dataset array
        idx_1 = np.where(y_clean == 1)[0]
        if len(idx_1) == 0:
            continue
            
        target_idx = idx_1[0]
        target_latent = X_clean[target_idx].reshape(1, -1)
        
        # 4. Measure Attack Success Rate (ASR)
        target_prediction = clf.predict(target_latent)[0]
        attack_success = 1 if target_prediction == 0 else 0
        
        # 5. Measure Clean Accuracy (Stealth)
        _, X_test_clean, _, y_test_clean = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
        )
        clean_acc = accuracy_score(y_test_clean, clf.predict(X_test_clean))
        
        results.append({
            'Dataset': dataset_name,
            'Method': method,
            'Rate': rate,
            'ASR': attack_success,
            'Clean Test Accuracy': clean_acc
        })
            
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No valid data found to plot. Check your CSV paths.")
        return
        
    # Aggregate results
    agg_df = results_df.groupby(['Method', 'Rate']).agg(
        Mean_ASR=('ASR', lambda x: np.mean(x) * 100), 
        Mean_Clean_Acc=('Clean Test Accuracy', lambda x: np.mean(x) * 100)
    ).reset_index()
    
    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Attack Success Rate (ASR)
    sns.lineplot(
        data=agg_df, x='Rate', y='Mean_ASR', hue='Method', 
        marker='o', linewidth=3, markersize=10, ax=ax1
    )
    ax1.set_title("Attack Success Rate (MLP Target Model)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Poisoning Rate", fontsize=12)
    ax1.set_ylabel("ASR (%)", fontsize=12)
    ax1.set_ylim(-5, 105)
    
    # Plot 2: Clean Test Accuracy (Stealth)
    sns.lineplot(
        data=agg_df, x='Rate', y='Mean_Clean_Acc', hue='Method', 
        marker='s', linewidth=2, linestyle='--', ax=ax2
    )
    ax2.set_title("MLP Overall Accuracy (Stealth)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Poisoning Rate", fontsize=12)
    ax2.set_ylabel("Overall Accuracy (%)", fontsize=12)
    ax2.set_ylim(85, 100) 
    
    plt.tight_layout()
    save_path = "attack_asr_mlp_evaluation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Evaluation complete. Plot saved to {save_path}")

if __name__ == "__main__":
    evaluate_attack_success_rate_mlp(meta_db_path="data/meta_db_image.csv")
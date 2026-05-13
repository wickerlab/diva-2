import os
import torch
import torchvision.models as models
import itertools
import pandas as pd
import logging
import random
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logger = logging.getLogger("ImageFetcher")
import glob
import itertools

def fetch_and_binarize_images(sources, n_max, base_folder="data", db_path=None, max_pair=10):
    logger.info("Fetching Datasets from Local Raw Downloads...")
    
    image_dir = os.path.join(base_folder, "raw_images")
    clean_dir = os.path.join(base_folder, "clean_data")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    processed_datanames = set()
    if db_path and os.path.exists(db_path):
        try:
            df = pd.read_csv(db_path)
            processed_datanames = set(df['Data'].values)
        except Exception: pass

    # Setup feature extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    latent_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()

    count_selected = 0
    MAX_PAIRS_PER_SOURCE = max_pair
    MAX_POINTS_NEEDED = 5000

    # Scan for locally downloaded raw pt files
    downloaded_files = glob.glob(os.path.join(image_dir, "*_full.pt"))

    for full_pt_path in downloaded_files:
        if count_selected >= n_max:
            break
            
        safe_name = os.path.basename(full_pt_path).replace("hf_", "").replace("_full.pt", "")
        
        # Check if we've already maxed out pairs for this source
        existing_count = sum(1 for d in processed_datanames if d.startswith(f"hf_{safe_name}_") or d.startswith(f"{safe_name}_"))
        if existing_count > 0:
            logger.info(f"Dataset '{safe_name}' already present in MetaDB. Skipping local load.")
            continue
            
        try:
            logger.info(f"Processing local dataset: {safe_name}")
            
            # Load pre-downloaded massive dataset
            full_data = torch.load(full_pt_path)
            X_all, y_all = full_data["X"], full_data["y"]
            
            unique_classes = torch.unique(y_all).tolist()
            if len(unique_classes) < 2: continue

            valid_pairs = []
            all_combinations = list(itertools.combinations(unique_classes, 2))
            
            # Find all combinations not yet in the DB
            for c0, c1 in all_combinations:
                dataname = f"hf_{safe_name}_{c0}_vs_{c1}"
                if dataname not in processed_datanames:
                    valid_pairs.append((c0, c1, dataname))

            random.shuffle(valid_pairs)
            valid_pairs = valid_pairs[:MAX_PAIRS_PER_SOURCE]

            if not valid_pairs: continue

            for c0, c1, dataname in valid_pairs:
                if count_selected >= n_max: break
                
                pt_file_path = os.path.join(image_dir, f"{dataname}.pt")
                csv_file_path = os.path.join(clean_dir, f"{dataname}_clean.csv")
                
                if not os.path.exists(pt_file_path):
                    # Filter for only the two target classes natively using PyTorch
                    mask = (y_all == c0) | (y_all == c1)
                    X_filtered = X_all[mask]
                    y_filtered = y_all[mask]
                    
                    if len(y_filtered) < 100: continue
                    
                    # Subsample if too large
                    if len(y_filtered) > MAX_POINTS_NEEDED:
                        indices = torch.randperm(len(y_filtered))[:MAX_POINTS_NEEDED]
                        X_filtered = X_filtered[indices]
                        y_filtered = y_filtered[indices]
                        
                    # Strict Binarization (0 for class c0, 1 for class c1)
                    y_bin = torch.where(y_filtered == c0, torch.tensor(0), torch.tensor(1))
                    
                    # Final subsample
                    max_n = min(len(y_bin), MAX_POINTS_NEEDED)
                    n_subsampling = torch.randint(low=max_n // 4, high=max_n, size=(1,)).item()
                    sub_indices = torch.randperm(len(y_bin))[:n_subsampling]
                    
                    torch.save({"X": X_filtered[sub_indices], "y": y_bin[sub_indices]}, pt_file_path)

                # Keep the same CSV saving approach
                if not os.path.exists(csv_file_path):
                    data = torch.load(pt_file_path)
                    X_images, y_labels = data["X"], data["y"]
                    latent_vectors = []
                    
                    with torch.no_grad():
                        for i in range(0, len(X_images), 128):
                            batch = X_images[i:i+128].to(device)
                            latent_vectors.append(latent_extractor(batch).squeeze().cpu())
                            
                    X_tab = torch.cat(latent_vectors).numpy()
                    
                    # Flatten dimensions in case shape is [N, 512, 1, 1]
                    if len(X_tab.shape) > 2:
                        X_tab = X_tab.reshape(X_tab.shape[0], -1)
                        
                    df = pd.DataFrame(X_tab, columns=[f"feature_{i}" for i in range(X_tab.shape[1])])
                    df['y'] = y_labels.numpy()
                    df.to_csv(csv_file_path, index=False)
                    
                count_selected += 1
                logger.info(f"    Prepared: {dataname} ({count_selected}/{n_max})")
                
                # Yield back to main.py orchestrator which automatically triggers poisoning & C-Measures
                yield csv_file_path

        except Exception as e:
            logger.warning(f"Failed to process local file {full_pt_path}: {e}")
            continue
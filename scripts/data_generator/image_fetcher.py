import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import itertools
import pandas as pd
import logging
import random
from datasets import load_dataset, Image

# --- NUKE HUGGING FACE SPAM ---
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error" 
os.environ["HF_HUB_MAX_RETRIES"] = "0"

logger = logging.getLogger("ImageFetcher")

def _process_single_hf_dataset(src, safe_name, image_dir, clean_dir, processed_datanames, device, latent_extractor, transform, max_pairs, max_points, current_count, n_max):
    """
    Helper function to process a single dataset. 
    Returns (generated_csvs, new_count, remove_flag)
    """
    pt_file_path = os.path.join(image_dir, f"hf_{safe_name}_full.pt")
    generated_csvs = []
    new_count = current_count
    remove_flag = False

    # =========================================================
    # PHASE A: Generate _full.pt if missing
    # =========================================================
    if not os.path.exists(pt_file_path):
        try:
            logger.info(f"Processing raw HF dataset: {src}")
            dataset = load_dataset(src, split="train", streaming=False)
            
            image_col = next((col for col, f in dataset.features.items() if isinstance(f, Image)), None)
            label_col = next((col for col in dataset.features.keys() if 'label' in col.lower() or 'class' in col.lower()), None)
            
            if not image_col or not label_col:
                logger.warning(f"[{src}] Could not auto-detect Image/Label columns. Flagging for removal.")
                return generated_csvs, new_count, True

            X_tensors = []
            y_tensors = []
            
            for item in dataset:
                try:
                    img_tensor = transform(item[image_col])
                    label = int(item[label_col])
                    X_tensors.append(img_tensor)
                    y_tensors.append(torch.tensor(label))
                except Exception: pass
                    
            if len(X_tensors) < 10:
                logger.warning(f"[{src}] Not enough valid images. Flagging for removal.")
                return generated_csvs, new_count, True
                
            X_all = torch.stack(X_tensors)
            y_all = torch.stack(y_tensors)
            torch.save({"X": X_all, "y": y_all}, pt_file_path)
            logger.info(f"✅ Generated local payload: {pt_file_path}")

        except Exception as e:
            logger.warning(f"Failed to process HF dataset {src}: {e}. Flagging for removal.")
            return generated_csvs, new_count, True

    # =========================================================
    # PHASE B: Binarize and Extract Latents
    # =========================================================
    try:
        full_data = torch.load(pt_file_path)
        X_all, y_all = full_data["X"], full_data["y"]
        
        unique_classes = torch.unique(y_all).tolist()
        if len(unique_classes) < 2: 
            logger.warning(f"[{src}] Less than 2 classes found. Cannot binarize. Flagging for removal.")
            return generated_csvs, new_count, True

        valid_pairs = []
        all_combinations = list(itertools.combinations(unique_classes, 2))
        
        for c0, c1 in all_combinations:
            dataname = f"hf_{safe_name}_{c0}_vs_{c1}"
            if dataname not in processed_datanames:
                valid_pairs.append((c0, c1, dataname))

        random.shuffle(valid_pairs)
        valid_pairs = valid_pairs[:max_pairs]

        for c0, c1, dataname in valid_pairs:
            if new_count >= n_max: break
            
            bin_pt_path = os.path.join(image_dir, f"{dataname}.pt")
            csv_file_path = os.path.join(clean_dir, f"{dataname}_clean.csv")
            
            if not os.path.exists(bin_pt_path):
                mask = (y_all == c0) | (y_all == c1)
                X_filtered = X_all[mask]
                y_filtered = y_all[mask]
                
                if len(y_filtered) < 100: continue
                
                if len(y_filtered) > max_points:
                    indices = torch.randperm(len(y_filtered))[:max_points]
                    X_filtered = X_filtered[indices]
                    y_filtered = y_filtered[indices]
                    
                y_bin = torch.where(y_filtered == c0, torch.tensor(0), torch.tensor(1))
                
                max_n = min(len(y_bin), max_points)
                n_subsampling = torch.randint(low=max_n // 4, high=max_n, size=(1,)).item()
                sub_indices = torch.randperm(len(y_bin))[:n_subsampling]
                
                torch.save({"X": X_filtered[sub_indices], "y": y_bin[sub_indices]}, bin_pt_path)

            if not os.path.exists(csv_file_path):
                data = torch.load(bin_pt_path)
                X_images, y_labels = data["X"], data["y"]
                latent_vectors = []
                
                with torch.no_grad():
                    for i in range(0, len(X_images), 128):
                        batch = X_images[i:i+128].to(device)
                        latent_vectors.append(latent_extractor(batch).flatten(1).cpu())
                        
                X_tab = torch.cat(latent_vectors).numpy()
                    
                df = pd.DataFrame(X_tab, columns=[f"feature_{i}" for i in range(X_tab.shape[1])])
                df['y'] = y_labels.numpy()
                df.to_csv(csv_file_path, index=False)
                
            new_count += 1
            generated_csvs.append(csv_file_path)
            logger.info(f"    Prepared: {dataname} ({new_count}/{n_max})")

    except Exception as e:
        logger.warning(f"Failed to binarize local file {pt_file_path}: {e}. Flagging for removal.")
        return generated_csvs, new_count, True

    # Return the clean list of paths and False for the remove_flag.
    return generated_csvs, new_count, False


def fetch_and_binarize_images(n_max, base_folder="data", db_path=None, max_pair=10, csv_path="data/hf_image_datasets.csv", max_size_gb=1.0):
    logger.info("Fetching and Processing Datasets from Hugging Face Cache...")
    
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

    # 1. Setup Models & Transforms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    latent_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 2. Load Registry
    if not os.path.exists(csv_path):
        logger.error(f"Cannot find HF CSV registry at {csv_path}. Please run the scraper first.")
        return []

    registry_df = pd.read_csv(csv_path)
    if 'Size_GB' in registry_df.columns:
        registry_df = registry_df[registry_df['Size_GB'] < max_size_gb]
        
    sources = registry_df["Dataset"].tolist()
    
    count_selected = 0
    MAX_PAIRS_PER_SOURCE = max_pair
    MAX_POINTS_NEEDED = 5000
    
    all_generated_files = []

    # 3. Process Sources Iteratively
    for src in sources:
        if count_selected >= n_max:
            break
            
        safe_name = src.replace("/", "_").lower()
        
        # Check if we've already generated data for this source in the MetaDB
        existing_count = sum(1 for d in processed_datanames if d.startswith(f"hf_{safe_name}_") or d.startswith(f"{safe_name}_"))
        if existing_count > 0:
            logger.info(f"Dataset '{safe_name}' already present in MetaDB. Skipping.")
            continue
            
        # Dispatch to the functionally scoped helper
        new_csvs, count_selected, remove_flag = _process_single_hf_dataset(
            src, safe_name, image_dir, clean_dir, processed_datanames, 
            device, latent_extractor, transform, 
            MAX_PAIRS_PER_SOURCE, MAX_POINTS_NEEDED, count_selected, n_max
        )
        
        # --- NEW: Immediate CSV cleanup if dataset failed ---
        if remove_flag:
            # Filter out the offending row
            registry_df = registry_df[registry_df["Dataset"] != src]
            # Immediately save the updated registry back to disk
            registry_df.to_csv(csv_path, index=False)
            logger.info(f"🗑️ Removed '{src}' from {csv_path} permanently.")
            
            # Optional: Clean up the corrupted .pt file if it was partially written
            pt_file_path = os.path.join(image_dir, f"hf_{safe_name}_full.pt")
            if os.path.exists(pt_file_path):
                os.remove(pt_file_path)
                logger.info(f"🧹 Cleaned up corrupted raw file: {pt_file_path}")
        else:
            all_generated_files.extend(new_csvs)

    return all_generated_files
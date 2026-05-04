import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import itertools
import pandas as pd
import logging
import random
import time

# NEW: Hugging Face Imports
from huggingface_hub import HfApi
from datasets import load_dataset, Image

logger = logging.getLogger("ImageFetcher")

def get_dynamic_image_sources(n_sources):
    """Fetches a list of dataset names from the Hugging Face Hub tagged for image classification."""
    logger.info(f"Querying Hugging Face API for top {n_sources} image classification datasets...")
    api = HfApi()
    
    # Filter for image classification tasks, sorted by downloads to get high-quality ones first
    datasets = api.list_datasets(
        filter="task_categories:image-classification", 
        sort="downloads", 
        direction=-1, 
        limit=n_sources * 3 # Pull extra in case some fail to load
    )
    
    return [d.id for d in datasets]

def fetch_and_binarize_images(sources, n_max, base_folder="data", db_path=None):
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

    # 1. Fetch Dynamic Sources if None are provided
    if not sources:
        sources = get_dynamic_image_sources(n_sources=15)

    # Standardize transforms for ResNet18
    # HF datasets load as PIL Images. We force RGB, resize to 32x32, and normalize.
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Standard ImageNet norm
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    latent_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()

    generated_csv_files = []
    count_selected = 0

    # 2. Iterate through dynamic datasets
    for src in sources:
        if count_selected >= n_max:
            break
            
        safe_name = src.replace("/", "_").lower()
        logger.info(f"Attempting to load dynamic dataset: {src}")
        
        try:
            # Load dataset (train split only to save time/space)
            dataset = load_dataset(src, split="train", streaming=False)
            
            # Autodetect columns (HF datasets name these inconsistently)
            image_col = next((col for col, f in dataset.features.items() if isinstance(f, Image)), None)
            label_col = next((col for col in dataset.features.keys() if 'label' in col.lower() or 'class' in col.lower()), None)
            
            if not image_col or not label_col:
                logger.warning(f"Could not auto-detect Image/Label columns for {src}. Skipping.")
                continue
                
            # Get unique classes
            unique_classes = set(dataset[label_col])
            if len(unique_classes) < 2:
                continue

           # Build binary pairs
            valid_pairs = []
            
            # 1. Get ALL possible combinations from the dataset (no longer limited to the first 10 classes)
            all_combinations = list(itertools.combinations(list(unique_classes), 2))
            
            for c0, c1 in all_combinations:
                dataname = f"hf_{safe_name}_{c0}_vs_{c1}"
                if dataname not in processed_datanames:
                    valid_pairs.append((c0, c1, dataname))

            max_pairs_per_source = 10
            random.shuffle(valid_pairs) # Randomize so we don't always get 0_vs_1
            valid_pairs = valid_pairs[:max_pairs_per_source]

            if not valid_pairs:
                logger.info(f"No new valid pairs found for {src} (maybe already processed). Skipping.")
                continue

            # Process pairs
            for c0, c1, dataname in valid_pairs:
                if count_selected >= n_max: 
                    break # Break out if we hit the global maximum
                
                pt_file_path = os.path.join(image_dir, f"{dataname}.pt")
                csv_file_path = os.path.join(clean_dir, f"{dataname}_clean.csv")
                
                if not os.path.exists(pt_file_path):
                    # Filter dataset to just the two classes
                    filtered_ds = dataset.filter(lambda x: x[label_col] in [c0, c1])
                    
                    if len(filtered_ds) < 100: # Skip datasets that are too tiny
                        continue
                        
                    # Process images to Tensors
                    X_tensors = []
                    y_tensors = []
                    
                    for item in filtered_ds:
                        try:
                            img_tensor = transform(item[image_col])
                            X_tensors.append(img_tensor)
                            # Binarize label
                            y_val = 0 if item[label_col] == c0 else 1
                            y_tensors.append(torch.tensor(y_val))
                        except Exception:
                            pass # Skip corrupted images
                            
                    if len(X_tensors) < 100: continue
                        
                    X_pair = torch.stack(X_tensors)
                    y_pair = torch.stack(y_tensors)

                    # Random subsampling
                    max_n = min(len(y_pair), 5000)
                    n_subsampling = torch.randint(low=max_n // 4, high=max_n, size=(1,)).item()
                    indices = torch.randperm(len(y_pair))[:n_subsampling]
                    
                    torch.save({"X": X_pair[indices], "y": y_pair[indices]}, pt_file_path)

                # --- LATENT EXTRACTION ---
                if not os.path.exists(csv_file_path):
                    data = torch.load(pt_file_path)
                    X_images, y_labels = data["X"], data["y"]
                    latent_vectors = []
                    
                    with torch.no_grad():
                        for i in range(0, len(X_images), 128):
                            batch = X_images[i:i+128].to(device)
                            latent_vectors.append(latent_extractor(batch).squeeze().cpu())
                            
                    X_tab = torch.cat(latent_vectors).numpy()
                    df = pd.DataFrame(X_tab, columns=[f"feature_{i}" for i in range(X_tab.shape[1])])
                    df['y'] = y_labels.numpy()
                    df.to_csv(csv_file_path, index=False)
                    
                generated_csv_files.append(csv_file_path)
                count_selected += 1
                logger.info(f"    Prepared: {dataname} ({count_selected}/{n_max})")

        except Exception as e:
            logger.warning(f"Failed to process dataset {src}: {e}")
            continue

    return generated_csv_files
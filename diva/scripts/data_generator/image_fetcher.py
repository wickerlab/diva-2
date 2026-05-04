import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import itertools
import pandas as pd
import logging
import random
import shutil

# Hugging Face Imports
from huggingface_hub import HfApi
from datasets import load_dataset, load_dataset_builder, Image, disable_progress_bar

disable_progress_bar()
logger = logging.getLogger("ImageFetcher")

def get_dynamic_image_sources(n_sources):
    logger.info(f"Querying Hugging Face API for top {n_sources} image classification datasets...")
    api = HfApi()
    datasets = api.list_datasets(
        filter="task_categories:image-classification", 
        sort="downloads",
        limit=100 + n_sources * 3
    )
    return [d.id for d in datasets]

def fetch_and_binarize_images(sources, n_max, base_folder="data", db_path=None, max_pair= 10):
    logger.info("Fetching Datasets (Streaming Mode)...")
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

    if not sources:
        sources = get_dynamic_image_sources(n_sources=n_max)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    latent_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()

    count_selected = 0
    MAX_PAIRS_PER_SOURCE = max_pair
    MAX_DOWNLOAD_SIZE_GB = 1.5
    MAX_POINTS_NEEDED = 5000

    for src in sources[100:]:
        if count_selected >= n_max:
            break
            
        safe_name = src.replace("/", "_").lower()
        
        existing_count = sum(1 for d in processed_datanames if d.startswith(f"hf_{safe_name}_"))
        if existing_count >= MAX_PAIRS_PER_SOURCE:
            logger.info(f"[{src}] Already has {existing_count} processed pairs. Skipping completely.")
            continue

        try:
            builder = load_dataset_builder(src)
            dl_size = builder.info.download_size or 0
            ds_size = builder.info.dataset_size or 0
            total_size_gb = (dl_size + ds_size) / (1024 * 1024 * 1024)
            
            if total_size_gb > MAX_DOWNLOAD_SIZE_GB:
                logger.warning(f"[{src}] Dataset too large (~{total_size_gb:.2f}GB). Skipping.")
                continue
            elif total_size_gb == 0:
                logger.warning(f"[{src}] Dataset size is unknown (missing metadata). Skipping to prevent massive blind downloads.")
                continue
            logger.info(f"Estimated size for {src}: {total_size_gb:.3f}GB")
        except Exception as e:
            continue
        
        temp_cache_dir = os.path.join(base_folder, "temp_hf_cache", safe_name)
        
        try:
            # Download to temp directory
            dataset = load_dataset(src, split="train", streaming=False, cache_dir=temp_cache_dir)
            
            image_col = next((col for col, f in dataset.features.items() if isinstance(f, Image)), None)
            label_col = next((col for col in dataset.features.keys() if 'label' in col.lower() or 'class' in col.lower()), None)
            
            if not image_col or not label_col:
                continue
                
            unique_classes = set(dataset[label_col])
            if len(unique_classes) < 2: continue

            valid_pairs = []
            all_combinations = list(itertools.combinations(list(unique_classes), 2))
            
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
                    filtered_ds = dataset.filter(lambda x: x[label_col] in [c0, c1])
                    if len(filtered_ds) < 100: continue
                    
                    if len(filtered_ds) > MAX_POINTS_NEEDED:
                        sub_indices = random.sample(range(len(filtered_ds)), MAX_POINTS_NEEDED)
                        filtered_ds = filtered_ds.select(sub_indices)
                        
                    X_tensors = []
                    y_tensors = []
                    
                    for item in filtered_ds:
                        try:
                            img_tensor = transform(item[image_col])
                            X_tensors.append(img_tensor)
                            y_val = 0 if item[label_col] == c0 else 1
                            y_tensors.append(torch.tensor(y_val))
                        except Exception: pass
                            
                    if len(X_tensors) < 100: continue
                        
                    X_pair = torch.stack(X_tensors)
                    y_pair = torch.stack(y_tensors)

                    max_n = min(len(y_pair), MAX_POINTS_NEEDED)
                    n_subsampling = torch.randint(low=max_n // 4, high=max_n, size=(1,)).item()
                    indices = torch.randperm(len(y_pair))[:n_subsampling]
                    
                    torch.save({"X": X_pair[indices], "y": y_pair[indices]}, pt_file_path)

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
                    
                count_selected += 1
                logger.info(f"    Prepared: {dataname} ({count_selected}/{n_max})")
                
                yield csv_file_path

        except Exception as e:
            logger.warning(f"Failed to process dataset {src}: {e}")
            continue
            
        finally:
            # --- CACHE CLEANUP ---
            if 'dataset' in locals():
                del dataset 
            if os.path.exists(temp_cache_dir):
                shutil.rmtree(temp_cache_dir)
                logger.info(f"    [Disk Management] Cleared temporary cache for {src}")
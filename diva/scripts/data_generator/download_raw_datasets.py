import os
# --- NUKE HUGGING FACE SPAM ---
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error" 
os.environ["HF_HUB_MAX_RETRIES"] = "0"

import torch
import torchvision.transforms as transforms
import pandas as pd
import logging
from tqdm import tqdm
from datasets import load_dataset, Image

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RawDataDownloader")
logging.getLogger("httpx").setLevel(logging.WARNING)

def download_all_raw_datasets(csv_path="data/hf_image_datasets.csv", base_folder="data", max_size_gb=1.0):
    image_dir = os.path.join(base_folder, "raw_images")
    os.makedirs(image_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        logger.error(f"Cannot find CSV registry at {csv_path}. Please run the scraper first.")
        return

    # Load the registry
    df = pd.read_csv(csv_path)
    
    # --- NEW: Filter for datasets strictly under 1GB ---
    if 'Size_GB' in df.columns:
        df = df[df['Size_GB'] < max_size_gb]
        
    sources = df["Dataset"].tolist()
    logger.info(f"Found {len(sources)} datasets under {max_size_gb}GB in registry. Starting bulk download...")

    # Standardize transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    for src in tqdm(sources, desc="Downloading & Processing"):
        safe_name = src.replace("/", "_").lower()
        pt_file_path = os.path.join(image_dir, f"hf_{safe_name}_full.pt")

        # --- ALREADY PRESENT: Skip if we already processed the full dataset ---
        if os.path.exists(pt_file_path):
            logger.info(f"[{src}] Already exists as {pt_file_path}. Skipping.")
            continue

        try:
            logger.info(f"Downloading: {src}")
            # Cache is kept permanently in Hugging Face's default location
            dataset = load_dataset(src, split="train", streaming=False)
            
            # Auto-detect Image and Label columns
            image_col = next((col for col, f in dataset.features.items() if isinstance(f, Image)), None)
            label_col = next((col for col in dataset.features.keys() if 'label' in col.lower() or 'class' in col.lower()), None)
            
            if not image_col or not label_col:
                logger.warning(f"[{src}] Could not auto-detect Image/Label columns. Skipping.")
                continue

            X_tensors = []
            y_tensors = []
            
            # Process the ENTIRE dataset without subsampling
            for item in dataset:
                try:
                    img_tensor = transform(item[image_col])
                    label = int(item[label_col])
                    
                    X_tensors.append(img_tensor)
                    y_tensors.append(torch.tensor(label))
                except Exception:
                    pass # Skip corrupted individual images
                    
            if len(X_tensors) < 10:
                logger.warning(f"[{src}] Not enough valid images processed. Skipping.")
                continue
                
            # Stack into massive tensors
            X_all = torch.stack(X_tensors)
            y_all = torch.stack(y_tensors)

            # Save the full dataset
            torch.save({"X": X_all, "y": y_all}, pt_file_path)
            logger.info(f"✅ Saved full dataset: {pt_file_path} (Size: {len(y_all)} images)")

        except Exception as e:
            logger.warning(f"Failed to process dataset {src}: {e}")
            continue

if __name__ == "__main__":
    download_all_raw_datasets()
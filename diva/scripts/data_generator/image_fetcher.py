import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import itertools
import pandas as pd
import logging

logger = logging.getLogger("ImageFetcher")

def fetch_and_binarize_images(n_max, base_folder="data", db_path=None):
    image_dir = os.path.join(base_folder, "raw_images")
    clean_dir = os.path.join(base_folder, "clean_data")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    # Prevent duplication by checking existing DB records
    processed_datanames = set()
    if db_path and os.path.exists(db_path):
        try:
            df = pd.read_csv(db_path)
            processed_datanames = set(df['Data'].values)
        except Exception: pass

    # Load ResNet for Latent Extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    latent_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()

    # Automatically set the correct stats and classes based on the dataset
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    dataset_cls = torchvision.datasets.CIFAR100
    num_classes = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    logger.info(f"Loading CIFAR...")
    dataset = dataset_cls(root=base_folder, train=True, download=True, transform=transform)
    
    # Scale and apply specific normalization manually for tensor storage
    X_all = torch.tensor(dataset.data).permute(0, 3, 1, 2).float() / 255.0
    X_all[..., 0, :, :] = (X_all[..., 0, :, :] - mean[0]) / std[0]
    X_all[..., 1, :, :] = (X_all[..., 1, :, :] - mean[1]) / std[1]
    X_all[..., 2, :, :] = (X_all[..., 2, :, :] - mean[2]) / std[2]
    
    y_all = torch.tensor(dataset.targets)
    
    binary_pairs = list(itertools.combinations(range(num_classes), 2))
    generated_csv_files = []
    count = 0

    for class_0, class_1 in binary_pairs:
        if count >= n_max: break
            
        dataname = f"CIFAR_{class_0}_vs_{class_1}"
        if dataname in processed_datanames:
            continue # Skip! Already in MetaDB.

        pt_file_path = os.path.join(image_dir, f"{dataname}.pt")
        csv_file_path = os.path.join(clean_dir, f"{dataname}_clean.csv")
        
        # 1. Generate binary raw images if missing
        if not os.path.exists(pt_file_path):
            mask = (y_all == class_0) | (y_all == class_1)
            X_pair, y_pair = X_all[mask], y_all[mask]
            y_pair = torch.where(y_pair == class_0, torch.tensor(0), torch.tensor(1))
            indices = torch.randperm(len(y_pair))[:2000]
            torch.save({"X": X_pair[indices], "y": y_pair[indices]}, pt_file_path)

        # 2. Generate Clean Latent CSV if missing
        if not os.path.exists(csv_file_path):
            logger.info(f"Extracting latent features for {dataname}...")
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
        count += 1
        logger.info(f"Prepared clean Latent CSV: {dataname} ({count}/{n_max})")

    return generated_csv_files
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import itertools
import pandas as pd
import logging
import random

logger = logging.getLogger("ImageFetcher")

# Centralized configuration for all vision datasets
SOURCE_CONFIG = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010),
        "cls": lambda r, t, d, tr: torchvision.datasets.CIFAR10(root=r, train=t, download=d, transform=tr),
        "num_classes": 10
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761),
        "cls": lambda r, t, d, tr: torchvision.datasets.CIFAR100(root=r, train=t, download=d, transform=tr),
        "num_classes": 100
    },
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970),
        "cls": lambda r, t, d, tr: torchvision.datasets.SVHN(root=r, split='train' if t else 'test', download=d, transform=tr),
        "num_classes": 10
    },
    "mnist": {
        "mean": (0.1307, 0.1307, 0.1307), "std": (0.3081, 0.3081, 0.3081),
        "cls": lambda r, t, d, tr: torchvision.datasets.MNIST(root=r, train=t, download=d, transform=tr),
        "num_classes": 10, "resize": True, "grayscale": True
    },
    "fashion_mnist": {
        "mean": (0.2860, 0.2860, 0.2860), "std": (0.3530, 0.3530, 0.3530),
        "cls": lambda r, t, d, tr: torchvision.datasets.FashionMNIST(root=r, train=t, download=d, transform=tr),
        "num_classes": 10, "resize": True, "grayscale": True
    },
    "textures": {
        "mean": (0.528, 0.474, 0.428), "std": (0.266, 0.255, 0.263),
        "cls": lambda r, t, d, tr: torchvision.datasets.DTD(root=r, split='train' if t else 'test', download=d, transform=tr),
        "num_classes": 47, "resize": True
    }
}

def fetch_and_binarize_images(sources, n_max, base_folder="data", db_path=None):
    image_dir = os.path.join(base_folder, "raw_images")
    clean_dir = os.path.join(base_folder, "clean_data")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    # 1. Prevent duplication by checking existing DB records
    processed_datanames = set()
    if db_path and os.path.exists(db_path):
        try:
            df = pd.read_csv(db_path)
            processed_datanames = set(df['Data'].values)
        except Exception: pass

    # 2. Build pools of valid tasks grouped by source
    available_pairs_by_source = {}
    total_available = 0
    
    for src in sources:
        if src not in SOURCE_CONFIG:
            logger.warning(f"Source '{src}' not found in configuration. Skipping.")
            continue
            
        num_classes = SOURCE_CONFIG[src]["num_classes"]
        valid_pairs_for_src = []
        
        for c0, c1 in itertools.combinations(range(num_classes), 2):
            dataname = f"{src}_{c0}_vs_{c1}"
            if dataname not in processed_datanames:
                valid_pairs_for_src.append((c0, c1, dataname))
                
        if valid_pairs_for_src:
            available_pairs_by_source[src] = valid_pairs_for_src
            total_available += len(valid_pairs_for_src)

    if total_available == 0:
        logger.info("All possible combinations for these sources already exist in the MetaDB!")
        return []

    # 3. Hierarchical Random Sampling (Equal probability for domains, then equal for classes)
    pairs_by_source = {}
    count_selected = 0
    
    if n_max > total_available:
        logger.info(f"Requested {n_max} datasets, but only {total_available} are left. Generating all remaining.")
        pairs_by_source = available_pairs_by_source
    else:
        while count_selected < n_max and available_pairs_by_source:
            # Step A: Pick a dataset source uniformly at random
            chosen_src = random.choice(list(available_pairs_by_source.keys()))
            
            # Step B: Pick a random un-processed class pair from that source
            pair_idx = random.randrange(len(available_pairs_by_source[chosen_src]))
            chosen_pair = available_pairs_by_source[chosen_src].pop(pair_idx)
            
            # Record it
            pairs_by_source.setdefault(chosen_src, []).append(chosen_pair)
            count_selected += 1
            
            # Step C: If a source runs out of combinations, remove it from the pool
            if not available_pairs_by_source[chosen_src]:
                del available_pairs_by_source[chosen_src]

    # 4. Load ResNet for Latent Extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    latent_extractor = torch.nn.Sequential(*(list(resnet.children())[:-1])).eval()

    generated_csv_files = []
    count = 0

    # 5. Process each source systematically
    for src, pairs in pairs_by_source.items():
        logger.info(f"Loading {src.upper()} dataset to process {len(pairs)} randomly chosen pair(s)...")
        cfg = SOURCE_CONFIG[src]
        
        # Build dynamic transform pipeline
        trans_list = []
        if cfg.get("resize"):
            trans_list.append(transforms.Resize((32, 32)))
        trans_list.append(transforms.ToTensor())
        if cfg.get("grayscale"):
            trans_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        trans_list.append(transforms.Normalize(cfg["mean"], cfg["std"]))
        
        transform = transforms.Compose(trans_list)
        
        # Load dataset
        dataset = cfg["cls"](r=base_folder, t=True, d=True, tr=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        X_all, y_all = next(iter(loader))
        
        for c0, c1, dataname in pairs:
            pt_file_path = os.path.join(image_dir, f"{dataname}.pt")
            csv_file_path = os.path.join(clean_dir, f"{dataname}_clean.csv")
            
            if not os.path.exists(pt_file_path):
                mask = (y_all == c0) | (y_all == c1)
                X_pair, y_pair = X_all[mask], y_all[mask]
                
                y_pair = torch.where(y_pair == c0, torch.tensor(0), torch.tensor(1))

                # Random subsampling to have randomly sized datasets
                max_n = min(len(y_pair), 5000)
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
                
            generated_csv_files.append(csv_file_path)
            count += 1
            logger.info(f"    Prepared: {dataname} ({count}/{count_selected})")

        # Memory Cleanup before moving to the next dataset
        del X_all, y_all, dataset, loader
        torch.cuda.empty_cache()

    return generated_csv_files
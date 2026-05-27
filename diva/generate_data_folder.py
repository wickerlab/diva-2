import os
import random
import numpy as np
import pandas as pd
import glob
import logging
from enum import Enum
from tqdm import tqdm
import concurrent.futures

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

# --- Modular Pipeline Imports ---
from scripts.data_generator.openml_fetcher import fetch_openml_datasets
from scripts.data_generator.image_fetcher import fetch_and_binarize_images

# --- Import Specific Poisoners directly from their modules ---
from scripts.poisoner.svm_poissvm.svm_poissvm_generate_metadb import PoisSVMPoisoner
from scripts.poisoner.svm_featurenoiseinjection.svm_featurenoiseinjection_generate_metadb import FeatureNoisePoisoner
from scripts.poisoner.svm_randomlabelflip.svm_randomlabelflip_generate_metadb import RandomFlipPoisoner
from scripts.poisoner.svm_alfa.svm_alfa_generate_metadb import AlfaPoisoner
from scripts.poisoner.svm_art.svm_art_generate_metadb import ArtSvmPoisoner
from scripts.poisoner.svm_feature_collision.svm_featurecollision import FeatureCollisionPoisoner
from scripts.witches_brew.witches_brew_generate_metadb import WitchesBrewPoisoner
from scripts.poisoner.poison_frogs.poison_frogs_generate_metadb import PoisonFrogsPoisoner
from scripts.poisoner.bullseye_polytope.bullseye_polytope_generate_metadb import BullseyePolytopePoisoner
from scripts.poisoner.badnets.badnet_generate_metadb import BadNetsPoisoner
from scripts.poisoner.learning_to_confuse.learning_to_confude_generate_metadb import AutoEncoderPoisoner
from scripts.poisoner.metapoison.metapoison_generate_metadb import MetaPoisonPoisoner


# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Standalone_DataGenerator")

# ==========================================
# Task Modality Configuration
# ==========================================
class TaskModality(str, Enum):
    TABULAR_BINARY = "tabular_binary"
    IMAGE_BINARY = "image_binary"
    IMAGE_MULTICLASS = "image_multiclass"

MODALITY_CONFIG = {
    TaskModality.TABULAR_BINARY: {
        "valid_poisoners": ["alfa_svm", "feature_noise_svm", "random_flip_svm", "feature_collision", "art_svm","pois_svm"]
    },
    TaskModality.IMAGE_BINARY: {
        "valid_poisoners": ["badnets", "autoencoder", "metapoison", "witches_brew", "poison_frogs", "random_flip_svm", "alfa_svm", "feature_noise_svm"] 
    }
}

POISONER_MAP = {
    "alfa_svm": AlfaPoisoner,
    "feature_noise_svm": FeatureNoisePoisoner,
    "random_flip_svm": RandomFlipPoisoner,
    "art_svm": ArtSvmPoisoner,
    "feature_collision": FeatureCollisionPoisoner,
    "witches_brew": WitchesBrewPoisoner,
    "poison_frogs": PoisonFrogsPoisoner,
    "bullseye_polytope": BullseyePolytopePoisoner,
    "badnets": BadNetsPoisoner,
    "autoencoder": AutoEncoderPoisoner,
    "metapoison": MetaPoisonPoisoner,
    "pois_svm": PoisSVMPoisoner,
}

def _execute_attack_task(method, file, rate, base_folder):
    """Worker function to execute a single attack in a separate process."""
    try:
        poisoner = POISONER_MAP[method](base_folder=base_folder)
        # We pass the rate as a single-item list since poisoners expect an iterable
        poisoner.apply_poisoning(file, [rate])
    except Exception as e:
        return f"Attack failed: {method} at {rate} on {file} - {e}"
    return None

# ==========================================
# Synthetic Data Generator Logic
# ==========================================
def generate_synthetic_data(n_sets, folder):
    """Generates a diverse set of clean synthetic base datasets."""
    N_SAMPLES_OPTIONS = np.arange(100, 1500, 100) 
    data_path = os.path.join(folder, "clean_data")
    os.makedirs(data_path, exist_ok=True)
    feature_ranges = list(range(20, 121, 20)) 
    
    grid = [] 
    for f in feature_ranges:
        grid.append({"n_samples": N_SAMPLES_OPTIONS, "n_classes": [2], "n_features": [f], "n_informative": [int(f * 0.7), int(f * 0.8), int(f * 0.9)], "n_repeated": [0], "weights": [[0.5, 0.5], [0.55, 0.45]], "flip_y": [0.0, 0.01], "class_sep": [1.5, 2.0, 2.5]})
        grid.append({"n_samples": N_SAMPLES_OPTIONS, "n_classes": [2], "n_features": [f], "n_informative": [int(f * 0.4), int(f * 0.5), int(f * 0.6)], "n_repeated": [0], "weights": [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]], "flip_y": [0.03, 0.05, 0.08], "class_sep": [0.8, 1.0, 1.2]})
        grid.append({"n_samples": N_SAMPLES_OPTIONS, "n_classes": [2], "n_features": [f], "n_informative": [int(f * 0.2), int(f * 0.3)], "n_repeated": [0], "weights": [[0.5, 0.5], [0.7, 0.3], [0.8, 0.2]], "flip_y": [0.10, 0.15, 0.20], "class_sep": [0.3, 0.5, 0.7]})

    param_sets = list(ParameterGrid(grid))
    selected_indices = np.random.choice(len(param_sets), n_sets, replace=(len(param_sets) < n_sets))
    generated_files = []

    for idx, i in enumerate(selected_indices):
        params = param_sets[i].copy()
        params["n_redundant"] = np.random.randint(0, max(1, params["n_features"] - params["n_informative"]))
        params["n_clusters_per_class"] = np.random.randint(1, 3) 
        params["random_state"] = np.random.randint(1000, 99999)

        X, y = make_classification(**params)
        X = StandardScaler().fit_transform(X)

        feature_names = [f"feature_{j}" for j in range(1, X.shape[1] + 1)]
        df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
        df["y"] = np.where(y > 0, 1, 0) 

        file_name = "f{:04d}_i{:03d}_r{:03d}_n{:04d}_sep{:.1f}".format(params["n_features"], params["n_informative"], params["n_redundant"], params["n_samples"], params["class_sep"])

        postfix = str(len(glob.glob(os.path.join(data_path, f"{file_name}_*.csv"))) + 1)
        output_path = os.path.join(data_path, f"{file_name}_{postfix}.csv")
        df.to_csv(output_path, index=False)
        generated_files.append(output_path)

    return generated_files

# ==========================================
# Orchestration Logic
# ==========================================
def apply_attacks_to_files(files, modality, n_attacks=4, base_folder="data"):
    """
    Randomly selects and applies n_attacks (method, rate) to a list of clean files.
    Executes the attacks for a single dataset in parallel.
    """
    config = MODALITY_CONFIG[modality]
    valid_methods = config["valid_poisoners"]
    
    advx_range = np.round(np.arange(0.05, 0.31, 0.05), 2)
    
    all_possible_attacks = [
        (m, r) for m in POISONER_MAP.keys() 
        if m in valid_methods 
        for r in advx_range
    ]
    
    if not files:
        logger.warning(f"No files provided for modality {modality.value}. Skipping attacks.")
        return

    logger.info(f"Applying {n_attacks} attacks to {len(files)} files for modality {modality.value}...")

    for file in tqdm(files, desc=f"Poisoning {modality.value}"):
        chosen_attacks = random.sample(all_possible_attacks, min(n_attacks, len(all_possible_attacks)))
        
        for method, rate in chosen_attacks:
            # Dispatch each chosen attack as a parallel background task
            try:
                _execute_attack_task(method, file, rate, base_folder)
            except Exception as error_msg:
                logger.error(error_msg)


def main():
    base_folder = "data"
    os.makedirs(base_folder, exist_ok=True)
    
    # Distribution Requirements
    n_synthetic = 250
    n_openml = 250
    n_hf_image = 467
    n_attacks_per_dataset = 4
    
    # ---------------------------------------------------------
    # 1. Synthetic Data (Tabular Binary)
    # ---------------------------------------------------------
    logger.info(f"--- Generating {n_synthetic} Synthetic Tabular Datasets ---")
    synthetic_files = generate_synthetic_data(n_synthetic, base_folder)
    apply_attacks_to_files(
        files=synthetic_files, 
        modality=TaskModality.TABULAR_BINARY, 
        n_attacks=n_attacks_per_dataset, 
        base_folder=base_folder
    )
    
    # ---------------------------------------------------------
    # 2. OpenML Data (Tabular Binary)
    # ---------------------------------------------------------
    logger.info(f"--- Fetching {n_openml} OpenML Tabular Datasets ---")
    openml_files = fetch_openml_datasets(
        n_max=n_openml, 
        folder=base_folder, 
        max_retries=3, 
        db_path=None 
    )
    apply_attacks_to_files(
        files=openml_files, 
        modality=TaskModality.TABULAR_BINARY, 
        n_attacks=n_attacks_per_dataset, 
        base_folder=base_folder
    )
    
    # ---------------------------------------------------------
    # 3. HF Image Data (Image Binary)
    # ---------------------------------------------------------
    logger.info(f"--- Fetching and Binarizing {n_hf_image} HF Image Datasets ---")
    hf_generator = fetch_and_binarize_images(
        n_max=n_hf_image, 
        base_folder=base_folder,
        max_pair=10
    )
    hf_files = list(hf_generator)
    
    apply_attacks_to_files(
        files=hf_files, 
        modality=TaskModality.IMAGE_BINARY, 
        n_attacks=n_attacks_per_dataset, 
        base_folder=base_folder
    )
    
    logger.info("=====================================================")
    logger.info("✅ Generation Phase Complete: 900 Datasets Generated & Poisoned.")
    logger.info("=====================================================")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()
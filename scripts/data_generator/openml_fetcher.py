import os
import numpy as np
import pandas as pd
import glob
import openml
import time
import hashlib
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample
import logging

logger = logging.getLogger("OpenMLFetcher")

def fetch_openml_datasets(n_max, folder, max_retries=3, db_path=None):
    """Fetches real-world binary classification datasets from OpenML, skipping existing ones."""
    logger.info(f"Fetching up to {n_max} OpenML binary datasets...")
    data_path = os.path.join(folder, "clean_data")
    os.makedirs(data_path, exist_ok=True)
    
    # --- ADDED: Read metadatabase to completely ignore already processed files ---
    processed_paths = set()
    if db_path and os.path.exists(db_path):
        try:
            processed_paths = set(pd.read_csv(db_path, usecols=['Path'])['Path'].values)
        except Exception:
            pass
    # ---------------------------------------------------------------------------
    
    # --- ADDED: Retry mechanism for the OpenML API ---
    datasets_df = None
    for attempt in range(max_retries):
        try:
            datasets_df = openml.datasets.list_datasets(output_format='dataframe')
            break # Success, break out of retry loop
        except Exception as e:
            wait_time = 5 * (attempt + 1)
            logger.warning(f"OpenML server error on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s... ({e})")
            time.sleep(wait_time)
            
    if datasets_df is None:
        logger.error("Failed to connect to OpenML after multiple attempts. Aborting OpenML fetch.")
        return []
    # --------------------------------------------------

    binary_datasets = datasets_df[
        (datasets_df['NumberOfClasses'] == 2) & 
        (datasets_df['NumberOfMissingValues'] == 0) &
        (datasets_df['NumberOfNumericFeatures'] > 5)
    ]
    
    generated_files = []
    count = 0
    
    for row in binary_datasets.itertuples():
        if count >= n_max: 
            break
            
        did = row.did
        dataset_name = row.name
        safe_name = str(dataset_name).lower().replace(' ', '_').replace('/', '')

        seed_val = int(hashlib.md5(safe_name.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed_val)
        
        existing_files = glob.glob(os.path.join(data_path, f"openml_{safe_name}_*.csv"))
        if existing_files:
            local_file = existing_files[0]
            
            # 1. If it is already fully processed in the metadatabase, ignore it entirely
            if local_file in processed_paths:
                continue 
                
            # 2. If it is downloaded locally but NOT in the metadatabase, reuse it!
            generated_files.append(local_file)
            count += 1
            logger.info(f"Dataset '{dataset_name}' already exists locally. Skipping download. ({count}/{n_max})")
            continue
            
        # --- ADDED: Inner retry mechanism for individual downloads ---
        for attempt in range(max_retries):
            try:
                dataset = openml.datasets.get_dataset(did)
                X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
                X_num = X.select_dtypes(include=['number']).dropna(axis=1)
                
                if X_num.shape[1] < 5: 
                    break # Not enough features, move to next dataset
                

                target_features = rng.randint(100, 1000) 
                target_samples = rng.randint(600, 6000)
                max_svd_components = X_num.shape[1] - 1
                actual_truncated = min(target_features, max_svd_components)
                
                if actual_truncated > 0 and X_num.shape[1] > actual_truncated + 1:
                    svd = TruncatedSVD(n_components=actual_truncated, random_state=42)
                    X_dense = svd.fit_transform(X_num)
                else:
                    if sp.issparse(X_num):
                        X_dense = X_num.toarray()
                    else:
                        X_dense = X_num.to_numpy() # Ensure dense numpy array
                
                if X_dense.shape[0] > target_samples:
                    X_dense, y = resample(X_dense, y, n_samples=target_samples, stratify=y, random_state=42)

                X_scaled = StandardScaler().fit_transform(X_dense)
                y_binary = pd.factorize(y)[0]
                
                df = pd.DataFrame(X_scaled, columns=[f"feature_{i}" for i in range(X_scaled.shape[1])], dtype=np.float32)
                df["y"] = y_binary
                
                file_name = f"openml_{safe_name}_n{len(y_binary)}_f{X_scaled.shape[1]}.csv"
                output_path = os.path.join(data_path, file_name)
                
                df.to_csv(output_path, index=False)
                generated_files.append(output_path)
                count += 1
                logger.info(f"Successfully loaded OpenML dataset '{dataset.name}' ({count}/{n_max})")
                break # Success, break out of retry loop
                
            except Exception as e:
                if "107" in str(e) or "server load" in str(e).lower():
                    logger.warning(f"OpenML server busy while downloading '{dataset_name}'. Retrying...")
                    time.sleep(3)
                else:
                    break # Not a server timeout error, just a weird dataset. Move on.
            
    return generated_files
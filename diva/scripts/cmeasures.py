import pandas as pd
import numpy as np
from pymfe.mfe import MFE
import concurrent.futures
from tqdm import tqdm
import logging
import os
import warnings

logger = logging.getLogger("CMeasures")

def _extract_single(args):
    file_path, features = args
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pd.read_csv(file_path)
            X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
            if len(X)>=20000:
                return None
            if len(X[0])>=10000:
                return None
            y = np.where(y == -1, 0, y)
            
            if features is None:
                mfe = MFE(groups=["complexity"], random_state=42)
            else:
                mfe = MFE(features=features, random_state=42)
                
            mfe.fit(X, y)
            f, v = mfe.extract()
            
        res = {"Path": file_path}
        res.update(dict(zip(f, v)))
        return res
    except Exception as e:
        logger.error(f"Error extracting from {file_path}: {e}")
        return None

def compute_cmeasures(file_paths, features=None, workers=None, db_path=None):
    """
    Takes a list of file paths, computes PyMFE complexities in parallel, 
    and returns a DataFrame. If db_path is provided, it skips files already in the DB.
    """
    paths_to_process = file_paths
    
    # Check the database and filter out paths that are already computed
    if db_path and os.path.exists(db_path):
        try:
            existing_db = pd.read_csv(db_path, usecols=['Path'])
            existing_paths = set(existing_db['Path'].values)
            paths_to_process = [p for p in file_paths if p not in existing_paths]
            
            skipped_count = len(file_paths) - len(paths_to_process)
            if skipped_count > 0:
                logger.info(f"Skipping {skipped_count} files (C-Measures already exist in DB).")
        except Exception as e:
            logger.warning(f"Could not read DB to filter paths: {e}. Proceeding with all.")
            
    if not paths_to_process:
        logger.info("All required C-Measures are already in the DB. Skipping extraction.")
        return pd.DataFrame() # Return empty DataFrame so append_to_db knows to do nothing

    logger.info(f"Extracting C-Measures for {len(paths_to_process)} new files...")
    results = []
    
    extraction_args = [(f, features) for f in paths_to_process]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract_single, arg): arg for arg in extraction_args}
        pbar = tqdm(concurrent.futures.as_completed(futures), total=len(extraction_args), desc="C-Measures")
        for future in pbar:
            original_args = futures[future]
            file_path = original_args[0]
            filename = os.path.basename(file_path)
            pbar.set_postfix(file=filename)

            result = future.result()
            if result is not None:
                results.append(result)
    return pd.DataFrame(results)
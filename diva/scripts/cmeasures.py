import pandas as pd
import numpy as np
from pymfe.mfe import MFE
import concurrent.futures
from tqdm import tqdm
import logging
import os

logger = logging.getLogger("CMeasures")

def _extract_single(args):
    file_path, features = args
    try:
        data = pd.read_csv(file_path)
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        y = np.where(y == -1, 0, y)
        
        if features is None:
            mfe = MFE(groups=["complexity"])
        else:
            mfe = MFE(features=features)
            
        mfe.fit(X, y)
        f, v = mfe.extract()
        
        res = {"Path": file_path}
        res.update(dict(zip(f, v)))
        return res
    except Exception as e:
        logger.error(f"Error extracting from {file_path}: {e}")
        return {"Path": file_path, "error": str(e)}

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
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(extraction_args), desc="C-Measures"):
            results.append(future.result())
            
    return pd.DataFrame(results)
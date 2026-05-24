import os
import logging
import warnings
import numpy as np
import pandas as pd
import concurrent.futures

from tqdm import tqdm
from pymfe.mfe import MFE
import pyarrow.csv as pv

logger = logging.getLogger("CMeasures")

def table_to_numpy(table):
    """
    Convert PyArrow table -> NumPy array efficiently.
    Assumes numeric data.
    """
    cols = [col.to_numpy(zero_copy_only=False) for col in table.columns]
    return np.column_stack(cols)


def _extract_single(args):
    """
    Process a single file using fast PyArrow reading and early pruning.
    """
    file_path, features, groups = args
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Fast CSV reading with PyArrow instead of Pandas
            table = pv.read_csv(
                file_path,
                read_options=pv.ReadOptions(use_threads=True),
                parse_options=pv.ParseOptions(delimiter=","),
            )

            data = table_to_numpy(table)

            if data.shape[0] == 0:
                return None

            X = data[:, :-1].astype(np.float32, copy=False)
            y = data[:, -1]

            # Early pruning (cheap checks to skip massive datasets)
            if X.shape[0] >= 20000 or X.shape[1] >= 10000:
                return None

            # Normalize labels
            y = np.where(y == -1, 0, y)

            # Initialize MFE
            if features is not None:
                mfe = MFE(features=features, random_state=42)
            else:
                mfe = MFE(groups=groups, random_state=42)

            # Compute MFE
            mfe.fit(X, y)
            f, v = mfe.extract()

        res = {"Path": file_path}
        res.update(dict(zip(f, v)))
        return res
        
    except Exception as e:
        logger.error(f"Error extracting from {file_path}: {e}")
        return None


def compute_cmeasures(file_paths, features=None, groups=None, workers=None, db_path=None):
    """
    Compute PyMFE complexity measures in parallel.

    - Uses PyArrow for fast CSV reading
    - Processes each file individually (no batching)
    - Skips already processed files if DB provided
    """
    paths_to_process = file_paths

    # Check for existing records in the database
    if db_path and os.path.exists(db_path):
        try:
            existing_paths = set(
                pd.read_csv(db_path, usecols=["Path"])["Path"].values
            )
            paths_to_process = [p for p in file_paths if p not in existing_paths]

            skipped = len(file_paths) - len(paths_to_process)
            if skipped > 0:
                logger.info(f"Skipping {skipped} files (already processed).")

        except Exception as e:
            logger.warning(f"Failed reading DB: {e}")

    if not paths_to_process:
        logger.info("All required C-Measures are already in the DB. Skipping extraction.")
        return pd.DataFrame()

    cpu_count = os.cpu_count() or 1
    workers = workers or max(1, cpu_count // 2)

    logger.info(f"Processing {len(paths_to_process)} files with {workers} workers...")

    # Default groups if neither features nor groups are specified
    if features is None and groups is None:
        groups = ["complexity", "model-based", "landmarking"]

    # Prepare arguments for single-file processing
    args_list = [(p, features, groups) for p in paths_to_process]
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit single files instead of batches
        futures = [executor.submit(_extract_single, arg) for arg in args_list]

        with tqdm(total=len(futures), desc="C-Measures") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    # Only append if a valid dictionary was returned (not None)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                
                pbar.update(1)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def add_new_measures_to_db(db_path, groups, workers=None):
    """
    Reads all paths from the existing DB, computes new PyMFE groups for them,
    and merges the new columns into the DB safely in batches.
    """
    logger.info(f"Adding new PyMFE groups {groups} to {db_path}...")

    if not os.path.exists(db_path):
        logger.error(f"Database {db_path} not found.")
        return

    df = pd.read_csv(db_path)
    if 'Path' not in df.columns:
        logger.error("The database does not contain a 'Path' column.")
        return

    file_paths = df['Path'].dropna().unique().tolist()

    extraction_args = [(f, None, groups) for f in file_paths]
    results = []

    # Calculate batch size dynamically based on workers
    actual_workers = workers if workers is not None else (os.cpu_count() or 4)
    batch_size = 4 * actual_workers

    # Set 'Path' as index so we can update specific rows easily
    df.set_index('Path', inplace=True)
    total_new_cols_added = set()

    def save_batch(batch_results):
        nonlocal df
        if not batch_results:
            return

        new_df = pd.DataFrame(batch_results)
        # Drop duplicates and set index to match main df
        new_df = new_df.drop_duplicates(subset=['Path'])
        new_df.set_index('Path', inplace=True)

        # Identify completely new columns that aren't in the main DB yet
        cols_to_add = new_df.columns.difference(df.columns).tolist()
        if cols_to_add:
            # Initialize new columns with None to avoid fragmentation
            df[cols_to_add] = None
            total_new_cols_added.update(cols_to_add)

        # Elegantly overwrite/update the specific rows with the newly computed features
        df.update(new_df)

        # Save back to CSV (resetting index puts 'Path' back as a normal column)
        df.reset_index().to_csv(db_path, index=False)
        logger.info(f"💾 Checkpoint saved: Processed and merged batch of {len(batch_results)} files.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract_single, arg): arg for arg in extraction_args}
        pbar = tqdm(concurrent.futures.as_completed(futures), total=len(extraction_args), desc="New Measures")

        for future in pbar:
            original_args = futures[future]
            pbar.set_postfix(file=os.path.basename(original_args[0]))

            result = future.result()
            if result is not None:
                results.append(result)

            # Trigger intermediate save when buffer is full
            if len(results) >= batch_size:
                save_batch(results)
                results = [] # Clear the buffer

    # Save any remaining results in the buffer
    if results:
        save_batch(results)

    logger.info(f"✅ Successfully computed {len(total_new_cols_added)} new features and completely updated {db_path}.")

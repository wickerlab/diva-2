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

def chunkify(lst, n_chunks):
    """Split list into roughly equal chunks."""
    n_chunks = max(1, n_chunks)
    k = max(1, len(lst) // n_chunks)
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def table_to_numpy(table):
    """
    Convert PyArrow table -> NumPy array efficiently.
    Assumes numeric data.
    """
    cols = [col.to_numpy(zero_copy_only=False) for col in table.columns]
    return np.column_stack(cols)

def _extract_batch(batch):
    """
    Process a batch of files inside a single worker.
    """
    features = batch[0][1]
    results = []

    if features is None:
        mfe = MFE(groups=["complexity"], random_state=42)
    else:
        mfe = MFE(features=features, random_state=42)

    for file_path, _ in batch:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                table = pv.read_csv(
                    file_path,
                    read_options=pv.ReadOptions(use_threads=True),
                    parse_options=pv.ParseOptions(delimiter=","),
                )

                data = table_to_numpy(table)

                if data.shape[0] == 0:
                    continue

                X = data[:, :-1].astype(np.float32, copy=False)
                y = data[:, -1]

                # Early pruning (cheap checks)
                if X.shape[0] >= 20000 or X.shape[1] >= 10000:
                    continue

                # Normalize labels
                y = np.where(y == -1, 0, y)

                # Compute MFE
                mfe.fit(X, y)
                f, v = mfe.extract()

                res = {"Path": file_path}
                res.update(dict(zip(f, v)))
                results.append(res)

        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {e}")

    return results

def compute_cmeasures(file_paths, features=None, workers=None, db_path=None):
    """
    Compute PyMFE complexity measures in parallel (optimized).

    - Uses PyArrow for fast CSV reading
    - Uses batching to reduce process overhead
    - Skips already processed files if DB provided
    """

    paths_to_process = file_paths

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
        logger.info("Nothing to process.")
        return pd.DataFrame()

    cpu_count = os.cpu_count() or 1
    workers = workers or max(1, cpu_count // 2)

    logger.info(f"Processing {len(paths_to_process)} files with {workers} workers...")

    batches = chunkify([(p, features) for p in paths_to_process], workers * 2)

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_extract_batch, b) for b in batches]

        with tqdm(total=len(futures), desc="C-Measures") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_result = future.result()
                    if batch_result:
                        results.extend(batch_result)
                except Exception as e:
                    logger.error(f"Batch failed: {e}")

                pbar.update(1)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)
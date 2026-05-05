import os
import pandas as pd
import logging
from tqdm import tqdm
from huggingface_hub import HfApi
import datasets
from datasets import load_dataset_builder
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

# Silence HF progress bars during sizing
datasets.disable_progress_bar()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatasetSizer")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Helper function for the thread pool
def fetch_dataset_size(ds):
    ds_id = ds.id
    downloads = getattr(ds, 'downloads', 0) or 0

    try:
        builder = load_dataset_builder(ds_id)
        dl_size = builder.info.download_size or 0
        ds_size = builder.info.dataset_size or 0
        total_size_gb = (dl_size + ds_size) / (1024 * 1024 * 1024)
        logger.info(f"{total_size_gb:.3f}G")

        # STRICT FILTER: Must have a known size, and must be > 0
        if total_size_gb > 0:
            return {
                "Dataset": ds_id,
                "Size_GB": round(total_size_gb, 4),
                "Downloads": downloads
            }
    except Exception:
        pass # Skip datasets that are private, gated, or broken

    return None

def build_hf_dataset_csv(n_sources=1000, output_csv="data/hf_image_datasets.csv"):
    logger.info(f"Querying Hugging Face API for up to {n_sources} datasets...")
    api = HfApi(token=os.getenv("HF_TOKEN"))

    all_datasets = api.list_datasets(
        filter="task_categories:image-classification",
        sort="downloads",
        limit=n_sources
    )

    # Deduplicate and sort globally by downloads
    unique_datasets = {d.id: d for d in all_datasets}.values()
    sorted_datasets = sorted(unique_datasets, key=lambda d: getattr(d, 'downloads', 0) or 0, reverse=True)[:n_sources]

    records = []
    logger.info(f"Checking metadata sizes for {len(sorted_datasets)} datasets using 15 parallel threads...")

    # --- MULTI-THREADING MAGIC ---
    # We use 15 workers so we don't accidentally trigger a Hugging Face rate limit block (HTTP 429)
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        # Submit all dataset checking tasks to the thread pool
        futures = {executor.submit(fetch_dataset_size, ds): ds for ds in sorted_datasets}

        # as_completed yields them as soon as they finish, keeping the progress bar accurate
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Sizing Datasets"):
            result = future.result()
            if result is not None:
                records.append(result)

    df = pd.DataFrame(records)

    # Re-sort the final dataframe by downloads just in case threading mixed up the order
    df.sort_values(by="Downloads", ascending=False, inplace=True)

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ Saved {len(df)} verified, sized datasets to {output_csv}")

    return output_csv

if __name__ == "__main__":
    # Run this manually once to build your database!
    build_hf_dataset_csv(n_sources=5000)
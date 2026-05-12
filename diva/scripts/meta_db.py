import os
import glob
import re
import argparse
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import math
from scripts.cmeasures import compute_cmeasures, add_new_measures_to_db

# ==========================================
# Task Modality Configuration
# ==========================================
class TaskModality(str, Enum):
    TABULAR_BINARY = "tabular_binary"
    IMAGE_BINARY = "image_binary"
    IMAGE_MULTICLASS = "image_multiclass"

MODALITY_CONFIG = {
    TaskModality.TABULAR_BINARY: {
        "db_path": "data/meta_db_universal.csv",
    },
    TaskModality.IMAGE_BINARY: {
        "db_path": "data/meta_db_universal.csv",
    },
    TaskModality.IMAGE_MULTICLASS: {
        "db_path": "data/meta_db_universal.csv",
    }
}

logger = logging.getLogger("MetaDB")

def append_to_db(db_path, metadata_list, cmeasures_df):
    """
    Merges metadata (Method, Rate, Is_Poisoned) with the extracted C-Measures
    and appends them to the target database.
    """
    meta_df = pd.DataFrame(metadata_list)
    
    if meta_df.empty or cmeasures_df.empty:
        logger.warning("Empty metadata or cmeasures provided. Skipping DB update.")
        return

    # Merge on Path
    merged_df = pd.merge(meta_df, cmeasures_df, on="Path", how="inner")
    
    if os.path.exists(db_path):
        existing_db = pd.read_csv(db_path)
        # Prevent duplicate entries by checking existing paths
        existing_paths = set(existing_db['Path'])
        merged_df = merged_df[~merged_df['Path'].isin(existing_paths)]
        
        if not merged_df.empty:
            updated_db = pd.concat([existing_db, merged_df], ignore_index=True)
            updated_db.to_csv(db_path, index=False)
            logger.info(f"Appended {len(merged_df)} new records to {db_path}.")
        else:
            logger.info(f"All files already exist in {db_path}. No new records added.")
    else:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        merged_df.to_csv(db_path, index=False)
        logger.info(f"Created new database at {db_path} with {len(merged_df)} records.")


def sync_filesystem_to_metadb(db_path, folders_to_scan, workers=None):
    """
    Scans the provided folders for any clean or poisoned CSV datasets that are NOT 
    in the metadatabase. It extracts their metadata from the path, computes their 
    C-Measures in batches, and progressively appends them to the DB.
    """
    batch_size = workers*4
    existing_paths = set()
    if os.path.exists(db_path):
        try:
            # Normalize paths from DB to guarantee accurate matching (handles Windows \ vs Linux /)
            db_paths = pd.read_csv(db_path, usecols=['Path'])['Path'].dropna().values
            existing_paths = set(os.path.normpath(p) for p in db_paths)
        except Exception as e:
            logger.warning(f"Could not read existing DB paths: {e}")

    # 1. Gather all CSVs
    all_csvs = []
    for folder in folders_to_scan:
        search_pattern = os.path.join(folder, "**", "*.csv")
        all_csvs.extend(glob.glob(search_pattern, recursive=True))

    total_files_found = len(all_csvs)

    # 2. Filter out files that are already in the DB
    files_to_process = []
    for f in all_csvs:
        f_norm = os.path.normpath(f)
        if any(skip in f_norm for skip in ["metadbs", "complexity_measures", "meta_database", "meta_db"]):
            continue
        
        # Verify and skip if already in the metadb
        if f_norm not in existing_paths:
            files_to_process.append(f_norm)

    unprocessed_count = len(files_to_process)
    
    # --- Goal 3: Log total vs unprocessed ---
    logger.info(f"Scanned folders and found {total_files_found} total CSV files.")
    
    if not files_to_process:
        logger.info("No missing files found. The MetaDB is perfectly synced with the filesystem.")
        return

    logger.info(f"Found {unprocessed_count} unprocessed datasets on disk. Extracting metadata...")

    # 3. Extract Metadata
    metadata_list = []
    for f in files_to_process:
        path_obj = Path(f)
        stem = path_obj.stem
        parent_dir = path_obj.parent.name
        
        if "clean_data" in f:
            metadata_list.append({
                "Data": stem.replace("_clean", ""), 
                "Path": f,
                "Method": "clean",
                "Rate": 0.0,
                "Is_Poisoned": 0
            })
        elif "poisoned_data" in f:
            method = parent_dir 
            rate_match = re.search(r'_([0-9]+\.[0-9]+)$', stem)
            if rate_match:
                rate = float(rate_match.group(1))
                base_without_rate = stem[:rate_match.start()]
                
                dataname = base_without_rate
                for method_variant in [method, method.replace("_svm", ""), method.replace("_", "")]:
                    if dataname.endswith(f"_{method_variant}"):
                        dataname = dataname[:-(len(method_variant)+1)]
                        break
                        
                metadata_list.append({
                    "Data": dataname,
                    "Path": f,
                    "Method": method,
                    "Rate": rate,
                    "Is_Poisoned": 1 if rate > 0 else 0
                })
            else:
                logger.warning(f"Could not parse rate from filename: {f}. Skipping.")
        else:
            logger.warning(f"Could not classify file origin (not in clean_data or poisoned_data): {f}")

    if not metadata_list:
        logger.info("No valid metadata could be parsed from the missing files.")
        return

    # 4. Process progressively in Batches
    logger.info(f"Computing C-Measures and syncing to DB in batches of {batch_size}...")
    
    total_batches = math.ceil(len(metadata_list) / batch_size)
    
    for i in range(0, len(metadata_list), batch_size):
        current_batch_num = (i // batch_size) + 1
        batch_meta = metadata_list[i : i + batch_size]
        paths_to_compute = [m["Path"] for m in batch_meta]
        
        logger.info(f"Processing batch {current_batch_num}/{total_batches} ({len(batch_meta)} files)...")
        
        # Compute measures for this specific chunk
        cmeasures_df = compute_cmeasures(paths_to_compute, workers=workers, db_path=db_path)
        
        # Append immediately to DB
        if not cmeasures_df.empty:
            append_to_db(db_path, batch_meta, cmeasures_df)
            
    logger.info("✅ Filesystem progressive sync complete.")


def print_db_statistics(db_path):
    """
    Reads the MetaDB, prints comprehensive statistics regarding dataset balance,
    and generates visual plots (bar chart for methods, heatmap for rates).
    """
    if not os.path.exists(db_path):
        logger.error(f"Database not found at: {db_path}")
        return
        
    try:
        df = pd.read_csv(db_path)
    except Exception as e:
        logger.error(f"Failed to read database: {e}")
        return

    db_dir = os.path.dirname(db_path)
    if not db_dir:
        db_dir = "."

    print("\n" + "="*50)
    print(f"📊 META-DATABASE STATISTICS: {os.path.basename(db_path)}")
    print("="*50)
    
    print(f"Total Datasets Processed: {len(df)}")
    
    if 'Is_Poisoned' in df.columns:
        print("\n--- ⚖️ Class & Domain Distribution ---")
        clean_df = df[df['Is_Poisoned'] == 0].copy()
        clean_count = len(clean_df)
        pois_count = len(df[df['Is_Poisoned'] == 1])
        
        print(f"Clean (0):    {clean_count} ({clean_count/len(df):.1%})")
        print(f"Poisoned (1): {pois_count} ({pois_count/len(df):.1%})")
        
        # Dynamically extract sources from clean datasets
        def extract_source(data_name):
            if data_name.startswith('f0') or data_name.startswith('f1'):
                return 'synthetic'
            return data_name.split('_')[0].lower()
            
        clean_df['Source'] = clean_df['Data'].apply(extract_source)
        source_counts = clean_df['Source'].value_counts()
        
        print("\nClean Baselines by Source:")
        for src, count in source_counts.items():
            print(f"  - {src.upper()}: {count} ({count/clean_count:.1%})")

    if 'Method' in df.columns:
        print("\n--- 🛡️ Breakdown by Method ---")
        method_counts = df['Method'].value_counts()
        print(method_counts.to_string())
        
        # Plot: Breakdown by Method (Fixed seaborn warning)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=method_counts.values, y=method_counts.index, hue=method_counts.index, palette='viridis', legend=False)
        plt.title(f"Dataset Breakdown by Method ({os.path.basename(db_path)})", fontweight='bold')
        plt.xlabel("Count")
        plt.ylabel("Method")
        plt.tight_layout()
        method_plot_path = os.path.join(db_dir, f"{os.path.splitext(os.path.basename(db_path))[0]}_method_dist.png")
        plt.savefig(method_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Saved method breakdown plot to: {method_plot_path}")

    if 'Method' in df.columns and 'Rate' in df.columns:
        print("\n--- 📈 Rate Distribution per Method ---")
        df_attacks = df[df['Method'] != 'clean'].copy()
        
        if not df_attacks.empty:
            df_attacks['Rate'] = df_attacks['Rate'].apply(lambda x: f"{x:.2f}")
            pivot = df_attacks.pivot_table(index='Method', columns='Rate', aggfunc='size', fill_value=0)
            
            # Plot: Heatmap of Rate Distribution
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Count'})
            plt.title(f"Attack Count Heatmap: Method vs. Rate ({os.path.basename(db_path)})", fontweight='bold')
            plt.ylabel("Poisoning Method")
            plt.xlabel("Poisoning Rate")
            plt.tight_layout()
            heatmap_plot_path = os.path.join(db_dir, f"{os.path.splitext(os.path.basename(db_path))[0]}_rate_heatmap.png")
            plt.savefig(heatmap_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"-> Saved rate heatmap to: {heatmap_plot_path}")
        else:
            print("No attack methods found in the database.")
            
    print("="*50 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(description="Meta-Database Utility Toolkit")
    parser.add_argument("action", choices=["sync", "stats", "add_measure"], help="Action to perform: 'sync' filesystem to DB, get DB 'stats', or 'add_measure' new PyMFE groups")
    
    # --- Modality and Overrides ---
    parser.add_argument("--modality", type=str, required=True, choices=[e.value for e in TaskModality], help="The core task modality to process.")
    parser.add_argument("--db_path", type=str, default=None, help="Override path to master DB")
    
    # Args for sync and extraction
    parser.add_argument("--folders", nargs='+', default=["data/clean_data", "data/poisoned_data"], help="Folders to scan for sync")
    parser.add_argument("--workers", type=int, default=None, help="Number of PyMFE workers")

    args = parser.parse_args()

    # Determine correct path from configuration
    config = MODALITY_CONFIG[TaskModality(args.modality)]
    db_path = args.db_path if args.db_path else config["db_path"]

    if args.action == "sync":
        sync_filesystem_to_metadb(db_path, folders_to_scan=args.folders, workers=args.workers)
    elif args.action == "stats":
        print_db_statistics(db_path)
    elif args.action == "add_measure":
        add_new_measures_to_db(
            db_path=db_path,
            groups=["model-based", "landmarking"],
            workers=args.workers
        )
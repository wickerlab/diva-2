import os
import glob
import re
import argparse
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.cmeasures import compute_cmeasures

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
    C-Measures, and securely appends them to the DB.
    """
    # 1. Load existing paths to avoid duplicates
    existing_paths = set()
    if os.path.exists(db_path):
        try:
            existing_paths = set(pd.read_csv(db_path, usecols=['Path'])['Path'].values)
        except Exception as e:
            logger.warning(f"Could not read existing DB paths: {e}")

    # 2. Find all CSVs in the specified directories
    all_csvs = []
    for folder in folders_to_scan:
        search_pattern = os.path.join(folder, "**", "*.csv")
        all_csvs.extend(glob.glob(search_pattern, recursive=True))

    # 3. Filter out system files, caches, and already processed files
    files_to_process = []
    for f in all_csvs:
        f_norm = os.path.normpath(f)
        
        # Skip internal system CSVs and C-Measure caches
        if any(skip in f_norm for skip in ["metadbs", "complexity_measures", "meta_database"]):
            continue
            
        if f_norm not in existing_paths:
            files_to_process.append(f_norm)

    if not files_to_process:
        logger.info("No missing files found. The MetaDB is perfectly synced with the filesystem.")
        return

    logger.info(f"Found {len(files_to_process)} unprocessed datasets on disk. Extracting metadata...")

    # 4. Reverse-engineer metadata from file paths
    metadata_list = []
    for f in files_to_process:
        path_obj = Path(f)
        stem = path_obj.stem
        parent_dir = path_obj.parent.name
        
        # Scenario A: It's a clean baseline dataset
        if "clean_data" in f:
            metadata_list.append({
                "Data": stem.replace("_clean", ""), 
                "Path": f,
                "Method": "clean",
                "Rate": 0.0,
                "Is_Poisoned": 0
            })
            
        # Scenario B: It's a poisoned dataset
        elif "poisoned_data" in f:
            method = parent_dir # The folder name is the method (e.g., 'alfa_svm')
            
            # Extract rate using regex (looks for _0.10 at the end of the filename)
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

    # 5. Compute C-Measures and Append
    logger.info(f"Computing C-Measures for {len(metadata_list)} recovered files...")
    paths_to_compute = [m["Path"] for m in metadata_list]
    
    cmeasures_df = compute_cmeasures(paths_to_compute, workers=workers, db_path=db_path)
    
    append_to_db(db_path, metadata_list, cmeasures_df)
    logger.info("✅ Filesystem sync complete.")


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

    # Set up a directory to save the plots (same folder as the database)
    db_dir = os.path.dirname(db_path)
    if not db_dir:
        db_dir = "."

    print("\n" + "="*50)
    print(f"📊 META-DATABASE STATISTICS: {os.path.basename(db_path)}")
    print("="*50)
    
    print(f"Total Datasets Processed: {len(df)}")
    
    if 'Is_Poisoned' in df.columns:
        print("\n--- ⚖️ Class Distribution ---")
        clean_count = len(df[df['Is_Poisoned'] == 0])
        openml_count = len(df[(df['Is_Poisoned'] == 0) & (df['Data'].str.contains("openml", na=False))])
        cifar_count = len(df[(df['Is_Poisoned'] == 0) & (df['Data'].str.contains("CIFAR", na=False))])
        pois_count = len(df[df['Is_Poisoned'] == 1])
        print(f"Clean (0):    {clean_count} ({clean_count/len(df):.1%})")
        print(f"Poisoned (1): {pois_count} ({pois_count/len(df):.1%})")
        print(f"Clean OpenML Dataset: {openml_count} ({openml_count/clean_count:.1%})")
        print(f"Clean CIFAR Dataset: {cifar_count} ({cifar_count/clean_count:.1%})")

    if 'Method' in df.columns:
        print("\n--- 🛡️ Breakdown by Method ---")
        method_counts = df['Method'].value_counts()
        print(method_counts.to_string())
        
        # Plot: Breakdown by Method
        plt.figure(figsize=(10, 6))
        sns.barplot(x=method_counts.values, y=method_counts.index, palette='viridis')
        plt.title(f"Dataset Breakdown by Method ({os.path.basename(db_path)})", fontweight='bold')
        plt.xlabel("Count")
        plt.ylabel("Method")
        plt.tight_layout()
        method_plot_path = os.path.join(db_dir, "db_method_distribution.png")
        plt.savefig(method_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> Saved method breakdown plot to: {method_plot_path}")

    if 'Method' in df.columns and 'Rate' in df.columns:
        print("\n--- 📈 Rate Distribution per Method ---")
        # Filter out the clean baseline to just see the attack distributions
        df_attacks = df[df['Method'] != 'clean'].copy()
        
        if not df_attacks.empty:
            breakdown = df_attacks.groupby(['Method', 'Rate']).size().reset_index(name='Count')
            print(breakdown.to_string(index=False))
            
            # Format Rate for cleaner plotting
            df_attacks['Rate'] = df_attacks['Rate'].apply(lambda x: f"{x:.2f}")
            
            # Pivot table to create the 2D grid for the heatmap
            pivot = df_attacks.pivot_table(index='Method', columns='Rate', aggfunc='size', fill_value=0)
            
            # Plot: Heatmap of Rate Distribution
            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Count'})
            plt.title(f"Attack Count Heatmap: Method vs. Rate ({os.path.basename(db_path)})", fontweight='bold')
            plt.ylabel("Poisoning Method")
            plt.xlabel("Poisoning Rate")
            plt.tight_layout()
            heatmap_plot_path = os.path.join(db_dir, "db_rate_heatmap.png")
            plt.savefig(heatmap_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"-> Saved rate heatmap to: {heatmap_plot_path}")
        else:
            print("No attack methods found in the database.")
            
    print("="*50 + "\n")


if __name__ == "__main__":
    # Setup CLI logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger("MetaDB")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s'))
        logger.addHandler(ch)

    parser = argparse.ArgumentParser(description="Meta-Database Utility Toolkit")
    parser.add_argument("action", choices=["sync", "stats"], help="Action to perform: 'sync' filesystem to DB, or get DB 'stats'")
    parser.add_argument("--db_path", type=str, default="data/universal_meta_database.csv", help="Target database path")
    
    # Args for sync
    parser.add_argument("--folders", nargs='+', default=["data/clean_data", "data/poisoned_data"], help="Folders to scan for sync")
    parser.add_argument("--workers", type=int, default=None, help="Number of PyMFE workers for sync")

    args = parser.parse_args()

    if args.action == "sync":
        sync_filesystem_to_metadb(args.db_path, folders_to_scan=args.folders, workers=args.workers)
    elif args.action == "stats":
        print_db_statistics(args.db_path)
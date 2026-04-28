import pandas as pd
import os
import logging

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
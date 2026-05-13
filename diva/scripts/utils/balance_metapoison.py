import pandas as pd
import random
import os
import argparse

# Import your existing pipeline modules
from scripts.poisoner.metapoison.metapoison_generate_metadb import MetaPoisonPoisoner
from scripts.cmeasures import compute_cmeasures
from scripts.meta_db import append_to_db

def main():
    parser = argparse.ArgumentParser(description="Balance MetaDB by adding MetaPoison attacks to the least-attacked HF datasets.")
    parser.add_argument("--db_path", type=str, default="data/meta_db_universal.csv", help="Path to your MetaDB CSV.")
    parser.add_argument("--base_folder", type=str, default="data", help="Base directory where raw/clean data is stored.")
    parser.add_argument("--num_datasets", type=int, default=100, help="Number of bottom datasets to augment with MetaPoison.")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for C-Measure extraction.")
    parser.add_argument("--min_rate", type=float, default=0.05, help="Minimum metapoison rate.")
    parser.add_argument("--max_rate", type=float, default=0.30, help="Maximum metapoison rate.")
    args = parser.parse_args()

    # 1. Load MetaDB
    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"Database not found at {args.db_path}")
        
    print(f"Loading database from {args.db_path}...")
    df = pd.read_csv(args.db_path)
    
    # 2. Filter datasets starting with 'hf_'
    hf_df = df[df['Data'].str.startswith("hf_")].copy()
    if hf_df.empty:
        print("No datasets starting with 'hf_' found in the database.")
        return
        
    # --- FIX 1: Normalize dataset names to fix the '_clean' mismatch ---
    # This ensures "hf_dataset" and "hf_dataset_clean" are grouped together
    hf_df['Normalized_Data'] = hf_df['Data'].apply(lambda x: x[:-6] if x.endswith('_clean') else x)
    
    # 3. Order by number of corresponding poisoning attacks using the normalized name
    clean_datasets = hf_df['Normalized_Data'].unique()
    
    # Count how many poisoned rows exist per normalized dataset
    poisoned_counts = hf_df[hf_df['Is_Poisoned'] == 1].groupby('Normalized_Data').size().to_dict()
    
    # Aggregate counts (ensure datasets with 0 attacks are also included)
    attack_counts = []
    for data_name in clean_datasets:
        attack_counts.append({
            'Normalized_Data': data_name,
            'attack_count': poisoned_counts.get(data_name, 0)
        })
        
    attack_counts_df = pd.DataFrame(attack_counts)
    
    # Sort ascending so the truly least attacked datasets are at the top
    attack_counts_df = attack_counts_df.sort_values(by='attack_count', ascending=True)
    
    print(f"\nTargeting {args.num_datasets} successfully augmented datasets...")
    
    # Extract the sorted list of normalized names
    least_attacked = attack_counts_df['Normalized_Data'].tolist()
    
    # Initialize Poisoner
    poisoner = MetaPoisonPoisoner(base_folder=args.base_folder)
    
    metadata_to_add = []
    new_csv_paths = []
    success_count = 0
    
    # 4. Add new poisoning attacks with a random rate
    for data_name in least_attacked:
        if success_count >= args.num_datasets:
            break
            
        # Find the clean data path for this dataset using Normalized_Data
        clean_rows = hf_df[(hf_df['Normalized_Data'] == data_name) & (hf_df['Is_Poisoned'] == 0)]
        if clean_rows.empty:
            continue
            
        clean_path = clean_rows.iloc[0]['Path']
        
        # Determine the expected raw tensor file path
        expected_pt_path = clean_path.replace("clean_data", "raw_images").replace("_clean.csv", ".pt")
        
        # --- FIX 2: Skip if the raw file does not exist to prevent the [Errno 2] crash ---
        if not os.path.exists(expected_pt_path):
            # Optionally print a debug statement: print(f"Skipping {data_name}: Missing raw file.")
            continue
        
        # Pick a random poisoning rate between min_rate and max_rate
        rate = round(random.uniform(args.min_rate, args.max_rate), 2)
        print(f"\nApplying MetaPoison to {data_name} with randomly chosen rate: {rate}")
        
        try:
            # apply_poisoning returns a list of dictionaries with metadata (Path, Rate, Method, etc.)
            metadata = poisoner.apply_poisoning(clean_path, [rate])
            metadata_to_add.extend(metadata)
            
            # Keep track of generated files for C-Measure step
            for m in metadata:
                new_csv_paths.append(m['Path'])
                
            success_count += 1
                
        except Exception as e:
            print(f"Failed to generate metapoison for {data_name}. Error: {e}")
            
    if not new_csv_paths:
        print("\nNo new attacks were successfully generated. Exiting.")
        return
        
    # 5. Compute C-Measures for the new datasets
    print(f"\nComputing C-Measures for {len(new_csv_paths)} newly generated files...")
    cmeasures_df = compute_cmeasures(new_csv_paths, workers=args.workers)
    
    # 6. Append back into the MetaDB Universal CSV
    if cmeasures_df is not None and not cmeasures_df.empty:
        print("Appending new measures and metadata to MetaDB...")
        append_to_db(args.db_path, metadata_to_add, cmeasures_df)
        print("✅ Successfully updated the database with new MetaPoison attacks!")
    else:
        print("❌ C-Measure extraction failed or returned empty. Database was not updated.")

if __name__ == "__main__":
    main()
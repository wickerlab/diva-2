import os
import glob
import pandas as pd
from abc import ABC, abstractmethod
from pymfe.mfe import MFE
import logging
from tqdm import tqdm
import concurrent.futures
from pathlib import Path

def _process_single_file(file_path):
    """
    Standalone helper function to extract MFE measures for a single file.
    Placed outside the class to ensure it is picklable by ProcessPoolExecutor.
    """
    try:
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        mfe = MFE(groups=["complexity"])
        mfe.fit(X, y)
        features, values = mfe.extract()

        result = {"file": os.path.basename(file_path)}
        result.update(dict(zip(features, values)))
        return result
        
    except Exception as e:
        # Catch errors so one corrupted file doesn't crash the whole parallel pool
        return {"file": os.path.basename(file_path), "error": str(e)}

class BasePoisoner(ABC):
    """
    Abstract Base Class for all poisoning methods.
    Handles standard directory setup, complexity extraction, and MetaDB merging.
    """
    def __init__(self, name, base_folder):
        self.logger = logging.getLogger(name)
        self.name = name
        self.base_folder = base_folder
        
        self.complexity_dir = os.path.join(base_folder, "poisoned_data", name)
        Path(self.complexity_dir).mkdir(parents=True, exist_ok=True)
        self.csv_score = os.path.join(base_folder, "poisoned_data", f"synth_{name}_score.csv")
        self.meta_db = os.path.join(base_folder, "metadbs", f"meta_database_{name}.csv")
        Path(os.path.join(base_folder, "metadbs")).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def apply_poisoning(self, file_path, advx_range):
        """
        MUST be implemented by child classes. 
        Contains the specific logic to poison the data and save the CSVs.
        """
        pass

    def extract_key(self, filename):
        """
        Extracts the core file name. Can be overridden by child classes if needed.
        """
        filename = os.path.basename(filename)
        return "_".join(filename.split("_")[:10])

    def extract_complexity_measures(self, max_workers=None):
        """
        Shared logic to extract MFE complexity measures using parallel processing.
        
        Args:
            max_workers (int): Number of parallel processes. If None, uses all available CPU cores.
        """
        poisoned_files = glob.glob(os.path.join(self.complexity_dir, "*.csv"))
        
        # Remove the output file from the list if it already exists
        output_path = os.path.join(self.complexity_dir, "complexity_measures.csv")
        if output_path in poisoned_files:
            poisoned_files.remove(output_path)

        results = []
        
        self.logger.info(f"Extracting complexity in parallel using up to {max_workers or 'all available'} workers...")

        # ProcessPoolExecutor is ideal for CPU-heavy mathematical tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files to the executor
            futures = {executor.submit(_process_single_file, file): file for file in poisoned_files}
            
            # as_completed yields futures as soon as they finish, allowing tqdm to update in real-time
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(poisoned_files), desc="Processing files"):
                result = future.result()
                
                # Check if our helper function caught an exception
                if "error" in result:
                    self.logger.error(f"Failed to process {result['file']}: {result['error']}")
                else:
                    results.append(result)
        
        # Dataframe creation and saving
        df_complexity = pd.DataFrame(results)
        df_complexity.to_csv(output_path, index=False)
        self.logger.info(f"Successfully saved complexity measures to {output_path}")

        return df_complexity
    
    def get_complexity_measures(self):
        """
        Retrieves complexity measures by loading a cached CSV.
        """
        output_path = os.path.join(self.complexity_dir, "complexity_measures.csv")

        # 1. Check if we have already processed these measures
        if os.path.exists(output_path):
            self.logger.info(f"Loading existing complexity measures from {output_path}")
            return pd.read_csv(output_path)
        else:
            raise ValueError("No complexity measures found, start by computing them.")

    def make_metadb(self, cmeasure_dataframe):
        """
        Shared logic to merge SVM scores with Complexity Measures.
        """
        if not os.path.exists(self.csv_score):
            self.logger.error(f"No CSV found at {self.csv_score}. Saving complexity data only.")
            cmeasure_dataframe.to_csv(self.meta_db, index=False)
            return

        csv_data = pd.read_csv(self.csv_score)
        
        # Apply key extraction
        csv_data["key"] = csv_data["Path.Poison"].apply(self.extract_key)
        cmeasure_dataframe["key"] = cmeasure_dataframe["file"].apply(self.extract_key)

        merged_data = pd.merge(csv_data, cmeasure_dataframe, on="key", how="inner")

        # Drop duplicates (Required by poissvm, safe for others)
        if "Path.Poison" in merged_data.columns:
            merged_data = merged_data.drop_duplicates(subset=["Path.Poison"])

        if merged_data.empty:
            self.logger.warning(f"Warning: No matching data found for merging in {self.name}.")

        merged_data.to_csv(self.meta_db, index=False)
        self.logger.info(f"Merged MetaDB saved to {self.meta_db}")

    def run_pipeline(self, file_paths, advx_range, entrypoint="poison", max_worker=None):
        """
        Executes the full pipeline for this specific poisoner.
        """
        self.logger.info(f"Starting Pipeline: {self.name.upper()}")
        os.makedirs(self.complexity_dir, exist_ok=True)
        cmeasure_df = None
        
        # 1. Apply Poisoning
        if entrypoint == "poison" :
            for (i,file) in enumerate(file_paths):
                self.logger.info(f"{i}/{len(file_paths)}: Started poisoning for file {file}")
                self.apply_poisoning(file, advx_range)

        if entrypoint in ("cmeasure", "poison") :
            # 2. Extract Complexity
            self.logger.info(f"\nExtracting complexity measures for {self.name}...")
            cmeasure_df = self.extract_complexity_measures(max_workers=max_worker)

        if entrypoint in ("metadb", "cmeasure", "poison") :
            if cmeasure_df is None:
                cmeasure_df = self.get_complexity_measures()

            # 3. Create MetaDB
            self.logger.info(f"\nCreating Meta Database for {self.name}...")
            self.make_metadb(cmeasure_df)
            return self.meta_db
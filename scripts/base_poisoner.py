import os
import logging

class BasePoisoner:
    def __init__(self, name, base_folder):
        self.name = name
        self.base_folder = base_folder
        self.poisoned_dir = os.path.join(base_folder, "poisoned_data", self.name)
        os.makedirs(self.poisoned_dir, exist_ok=True)
        self.logger = logging.getLogger(self.name.upper())

    def apply_poisoning(self, file_path, advx_range):
        """
        Subclasses MUST implement this.
        It should generate the poisoned CSVs and return a LIST OF DICTIONARIES containing the metadata.
        Example return: [{"Path": p, "Data": d, "Method": self.name, "Rate": r, "Is_Poisoned": 1}, ...]
        """
        raise NotImplementedError("Subclasses must implement apply_poisoning")
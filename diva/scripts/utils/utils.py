import datetime
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import random
import torch
import threading
import queue

logger = logging.getLogger(__name__)

class BackgroundPrefetcher:
    """
    Wraps any Python generator in a background thread.
    Downloads the next 'max_prefetch' items while the main thread is busy processing.
    """
    def __init__(self, generator, max_prefetch=3):
        self.generator = generator
        self.queue = queue.Queue(maxsize=max_prefetch)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for item in self.generator:
                self.queue.put(item) # Pauses here if queue is full
        except Exception as e:
            print(f"Background fetcher encountered an error: {e}")
        finally:
            self.queue.put(None) # Send termination signal

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

def log_cols(path_data):
    """Read data from a CSV file, output the column names"""
    df_data = pd.read_csv(path_data)
    cols = df_data.columns.tolist()

    for i, v in enumerate(cols):
        print(f"Column {i}: {v}")

def drop_cols(path_data, indices_to_drop):
    """Read data from a CSV file, drops columns by index, output the column names"""
    df_data = pd.read_csv(path_data)
    cols = df_data.columns.tolist()
    cols_to_drop = [cols[i] for i in indices_to_drop]
    df_data.drop(cols_to_drop, inplace=True, axis=1)
    return df_data


def time2str(time_elapsed):
    return time.strftime("%Hh%Mm%Ss", time.gmtime(time_elapsed))


def transform_label(y, target=-1):
    """Transform binary labels from {0, 1} to {-1, 1}. If target is 0, 
    transform them back to {0, 1}"""
    assert target in [-1, 0], f'Expect target to be -1 or 0, got {target}.'
    outputs = np.copy(y)
    neg_lbl = 0 if target == -1 else -1
    idx_neg_lbl = np.where(outputs == neg_lbl)[0]
    outputs[idx_neg_lbl] = target
    return outputs


def flip_binary_label(y, idx, use_neg_label=False):
    """Flip binary labels with given indices."""
    y_flip = np.copy(y)
    y_flip[y_flip == 0] = -1
    y_flip[idx] = - y_flip[idx]
    if use_neg_label:
        return y_flip
    y_flip[y_flip == -1] = 0
    return y_flip


def create_dir(path):
    """Create directory if the input path is not found."""
    if not os.path.exists(path):
        os.makedirs(path)


def open_csv(path_data, label_name='y'):
    """Read data from a CSV file, return X, y and column names."""
    df_data = pd.read_csv(path_data)
    y = df_data[label_name].to_numpy()
    df_data = df_data.drop([label_name], axis=1)
    cols = df_data.columns
    X = df_data.to_numpy()
    return X, y, cols


def to_csv(X, y, cols, path_data):
    """Save data into a CSV file."""
    logger.info(f'Save to: {path_data}')
    df = pd.DataFrame(X, columns=cols, dtype=np.float32)
    labels = len(np.unique(y))
    assert labels == 2, f'Expecting 2 classes, got {labels}'
    df['y'] = pd.Series(y, dtype=int)
    df['y'] = df['y'].astype('category').cat.codes
    df.to_csv(path_data, index=False)


def to_json(data_dict, path):
    """Save dictionary as JSON."""
    def converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

    with open(path, 'w') as file:
        json.dump(data_dict, file, default=converter)


def open_json(path):
    """Read JSON file."""
    try:
        with open(path, 'r') as file:
            data_json = json.load(file)
            return data_json
    except:
        print(f'Cannot open {path}')

def set_global_seed(seed: int = 42):
    """
    Locks down all sources of randomness for complete reproducibility.
    """
    # 1. Python built-in random module
    random.seed(seed)
    
    # 2. Numpy random state
    np.random.seed(seed)
    
    # 3. PyTorch random state (CPU and GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        
    # 4. CuDNN Determinism (Forces PyTorch to use deterministic algorithms)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 5. OS-level hash seed (for dictionary/set ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🌱 Global seed locked to: {seed}")
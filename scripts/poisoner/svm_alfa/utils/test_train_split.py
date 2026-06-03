"""
Split training and test sets (Copied synth)
"""
import os
from glob import glob
from pathlib import Path

from sklearn.model_selection import train_test_split

from utils.utils import create_dir, open_csv, to_csv


def test_train_split(path_data, path_output, test_size):
    filepath = str(Path(path_data).absolute())
    output = str(Path(path_output).absolute())

    create_dir(os.path.join(output, 'train'))
    create_dir(os.path.join(output, 'test'))

    path_list = sorted(glob(os.path.join(filepath, '*.csv')))
    print(f'Found {len(path_list)} datasets.')

    for p in path_list:
        X, y, cols = open_csv(p, label_name='y')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        dataname = Path(p).stem
        to_csv(X_train, y_train, cols, os.path.join(output, 'train', f'{dataname}_train.csv'))
        to_csv(X_test, y_test, cols, os.path.join(output, 'test', f'{dataname}_test.csv'))

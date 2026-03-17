import os
import glob
import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

def generate_datasets(num_datasets=100):
  N_SAMPLES = np.arange(1000, 3001, 200)
  N_CLASSES = 2  # Number of classes

  grid = []
  for f in range(4, 31):
    grid.append(
      {
        "n_samples": N_SAMPLES,
        "n_classes": [N_CLASSES],
        "n_features": [f],
        "n_repeated": [0],
        "n_informative": np.arange(f // 2, f + 1),
        "weights": [[0.4], [0.5], [0.6]],
      }
    )

  param_sets = list(ParameterGrid(grid))

  # Adjust redundant features and clusters per class for each parameter set
  for i in range(len(param_sets)):
    param_sets[i]["n_redundant"] = np.random.randint(
      0, high=param_sets[i]["n_features"] + 1 - param_sets[i]["n_informative"]
    )
    param_sets[i]["n_clusters_per_class"] = np.random.randint(
      1, param_sets[i]["n_informative"]
    )

  replace = len(param_sets) < num_datasets
  selected_indices = np.random.choice(len(param_sets), num_datasets, replace=replace)

  generated_dataframes = []  # Keep track of generated dataframes

  for i in selected_indices:
    param_sets[i]["random_state"] = np.random.randint(1000, np.iinfo(np.int16).max)

    X, y = make_classification(**param_sets[i])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    feature_names = ["x" + str(j) for j in range(1, X.shape[1] + 1)]
    df = pd.DataFrame(X, columns=feature_names, dtype=np.float32)
    df["y"] = y.astype(np.int32)  # Convert y to integers

    generated_dataframes.append(df)  # Store the dataframe

  return generated_dataframes  # Return the list of generated dataframes

def extract_complexity_measures(dataset):
  """Extract complexity measures from a single dataset (DataFrame)."""
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, -1].values

  # Initialize MFE (meta feature extractor) with complexity measures
  mfe = MFE(groups=["complexity"])

  # Fit the MFE model to the data
  mfe.fit(X, y)

  # Extract meta-features
  features, values = mfe.extract()

  # Collect results
  result = dict(zip(features, values))

  # Convert result to DataFrame
  results_df = pd.DataFrame([result])

  return results_df

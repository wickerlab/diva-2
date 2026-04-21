import argparse
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import warnings

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from pymfe.mfe import MFE
from aim import Run
from sklearn.model_selection import GridSearchCV

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MetaPipeline")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pymfe")

def generate_diverse_parameters(n_sets):
    """
    Generates highly diverse parameters using randomized continuous distributions
    to maximize the coverage of different C-Measures.
    """
    params_list = []
    for _ in range(n_sets):
        # Sample sizes from 300 to 5000
        n_samples = np.random.randint(300, 5001)
        # Features from 10 to 150
        n_features = np.random.randint(10, 151)
        
        # Distribute feature types randomly but mathematically soundly
        n_informative = np.random.randint(max(2, int(n_features * 0.1)), max(3, int(n_features * 0.95)))
        remaining = n_features - n_informative
        n_redundant = np.random.randint(0, remaining + 1)
        n_repeated = np.random.randint(0, remaining - n_redundant + 1)
        
        # --- FIX 1: Safely Randomize Classes ---
        # make_classification requires: n_classes * n_clusters_per_class <= 2**n_informative
        max_possible_classes = min(10, 2**n_informative) 
        # Generate between 2 and 9 classes, ensuring it doesn't break the hypercube limit
        n_classes = np.random.randint(2, max(3, max_possible_classes))
        
        # Shape and difficulty
        max_allowed_clusters = int((2**n_informative) / n_classes)
        # Cap at 3 to prevent overly complex dataset generation times
        max_clusters = max(1, min(3, max_allowed_clusters)) 
        n_clusters_per_class = np.random.randint(1, max_clusters + 1)
        
        class_sep = np.random.uniform(0.1, 3.5) # 0.1 is highly overlapping, 3.5 is perfectly separated
        flip_y = np.random.uniform(0.0, 0.3)    # Up to 30% label noise (simulates severe data corruption)
        
        # --- FIX 2: Dynamic Weights ---
        # Generate random weights for however many classes were selected, then normalize so they sum to 1.0
        raw_weights = np.random.uniform(0.1, 1.0, size=n_classes)
        weights = (raw_weights / raw_weights.sum()).tolist()
        
        params_list.append({
            "n_samples": n_samples,
            "n_classes": n_classes,
            "n_features": n_features,
            "n_informative": n_informative,
            "n_redundant": n_redundant,
            "n_repeated": n_repeated,
            "n_clusters_per_class": n_clusters_per_class,
            "class_sep": class_sep,
            "flip_y": flip_y,
            "weights": weights
        })
    return params_list

def build_meta_database(n_sets, output_dir):
    """Generates data, trains classifiers, calculates C-measures, and builds the MetaDB."""
    params_list = generate_diverse_parameters(n_sets)
    meta_records = []
    accuracies = []

    logger.info(f"Generating {n_sets} diverse datasets and computing C-Measures...")
    
    pbar = tqdm(params_list, desc="Processing Datasets")
    for idx, params in enumerate(pbar):
        try:
            # 1. Generate clean, highly randomized dataset
            X, y = make_classification(**params)
            
            # Enforce strict binary labels
            y = np.where(y > 0, 1, 0)
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            #! 2. Train a classifier on the train set and evaluate on test set
            clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
            clf.fit(X_train, y_train)
            test_accuracy = accuracy_score(y_test, clf.predict(X_test))
            pbar.set_postfix({'Test Accuracy': test_accuracy})
            accuracies.append(test_accuracy)

            # 3. Compute C-Measures strictly on the test set
            mfe = MFE(groups=["complexity"])
            mfe.fit(X_test, y_test)
            features, values = mfe.extract()

            # Compile record
            record = {"dataset_id": idx, "test_accuracy": test_accuracy}
            record.update(dict(zip(features, values)))
            meta_records.append(record)

        except Exception as e:
            logger.warning(f"Skipped dataset {idx} due to generation/MFE error: {e}")

    # Create DataFrame and save
    meta_df = pd.DataFrame(meta_records)
    
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "clean_meta_database.csv")
    meta_df.to_csv(db_path, index=False)
    logger.info(f"Meta-database saved to {db_path} with {len(meta_df)} valid records.")
    accuracies.sort()

    return meta_df, accuracies

def train_meta_learner(meta_df, accuracies, output_dir, total_datasets, description):
    """Trains an SVR and logs the run using Aim."""
    logger.info("Training Meta-Learner...")
    
    # Preprocess MetaDB
    # Fill missing PyMFE values with 0.0 to ensure consistent feature sizing
    meta_df = meta_df.fillna(0.0) 
    
    if meta_df.empty or len(meta_df) < 10:
        logger.error("Not enough data to train the meta-learner. Exiting.")
        return

    # Extract features (C-measures) and target (test accuracy)
    X_meta = meta_df.drop(columns=["dataset_id", "test_accuracy"]).values 
    y_meta = meta_df["test_accuracy"].values

    # Split meta-database for evaluation
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
        X_meta, y_meta, test_size=0.1, random_state=42
    )

    # --- Initialize Aim Run ---
    run = Run(experiment="MetaLearner_Training")
    run.set("description", description)
    run["hparams"] = {
        "output_dir": output_dir,
        "total_source_datasets": total_datasets,
        "valid_datasets_used": len(meta_df),
        "model_type": "SVR(kernel='rbf')"
    }
    for step, acc in enumerate(accuracies):
        run.track(acc, name="baseline_test_accuracies", step=step)

    try:
        # --- VERSION 1: TUNED SVR ---
        logger.info("Starting Grid Search for SVR hyperparameter tuning...")
        
        # Define the base pipeline
        base_pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
        
        # Define the parameter grid (note the 'svr__' prefix to target the SVR step in the pipeline)
        param_grid = {
            'svr__C': [0.1, 1.0, 10.0, 100.0],
            'svr__gamma': ['scale', 'auto', 0.01, 0.1],
            'svr__epsilon': [0.01, 0.05, 0.1, 0.2]
        }
        
        # Run 5-fold cross-validated grid search
        grid_search = GridSearchCV(base_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_m_train, y_m_train)
        
        # Extract the best model
        meta_learner = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logger.info(f"Best SVR Parameters found: {best_params}")
        
        # Log best params to Aim
        run["hparams"].update(best_params)

        # Evaluate
        y_m_pred = meta_learner.predict(X_m_test)
        mse = mean_squared_error(y_m_test, y_m_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_m_test, y_m_pred)
        r2 = r2_score(y_m_test, y_m_pred)
        
        logger.info(f"Meta-Learner Training Complete. Evaluation on {len(y_m_test)} test datasets:")
        logger.info(f"  - R² Score: {r2:.4f} (Higher is better, max 1.0)")
        logger.info(f"  - RMSE:     {rmse:.4f} (Lower is better)")
        logger.info(f"  - MAE:      {mae:.4f} (Lower is better)")
        logger.info(f"  - MSE:      {mse:.4f} (Lower is better)")

        # Track metrics in Aim
        run.track(mse, name='mean_squared_error', context={'subset': 'meta_test'})
        run.track(rmse, name='root_mean_squared_error', context={'subset': 'meta_test'})
        run.track(mae, name='mean_absolute_error', context={'subset': 'meta_test'})
        run.track(r2, name='r2_score', context={'subset': 'meta_test'})

        # Save Model
        model_path = os.path.join(output_dir, "metalearner_clean_svr.pkl")
        joblib.dump(meta_learner, model_path)
        logger.info(f"Meta-Learner model saved to {model_path}")

    except Exception as e:
        logger.exception("An error occurred during Meta-Learner training.")
    finally:
        # Ensure Aim run closes properly
        run.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Meta-Learner Pipeline")
    parser.add_argument("-n", "--nSets", default=500, type=int, help="Number of synthetic datasets to generate.")
    parser.add_argument("-o", "--output", default="./data", type=str, help="Output directory for DB and models.")
    parser.add_argument("-d", "--description", default="", type=str, help="Description of the run for aim.")
    args = parser.parse_args()

    # Step 1 & 2: Generate Data, Evaluate Classifier, Extract C-Measures -> MetaDB
    meta_dataframe, accuracies = build_meta_database(n_sets=args.nSets, output_dir=args.output)

    # Step 3: Train Meta-Learner and track with Aim
    train_meta_learner(meta_dataframe, accuracies, output_dir=args.output, total_datasets=args.nSets, description=args.description)
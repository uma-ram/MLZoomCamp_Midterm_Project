# Import Libraries

import pickle

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import StandardScaler

from custom_transformer import DictVectorizerTransformer

   
# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
def load_data():
    data  = pd.read_csv('./data/insurance.csv')
    return data

# ---------------------------------------------------------
# Evaluate Model
# ---------------------------------------------------------
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n=== Model Evaluation ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.2f}")

    return rmse, mae, r2


# ---------------------------------------------------------
# Train Final Model
# ---------------------------------------------------------
def train():
    
    print("Loading dataset...")
    df = load_data()

    # Target column
    target = "charges"

    # Features
    X = df.drop(columns=[target])
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=25)
    
    print("Shape:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)

    # ---------------------------------------------------------
    # Preprocessor (DV + Scaler)
    # ---------------------------------------------------------
    preprocessor = Pipeline([
        ("dv", DictVectorizerTransformer()),
        ("scaler", StandardScaler())
    ])

    # ---------------------------------------------------------
    # BEST Model (use your tuned parameters)
    # ---------------------------------------------------------
    best_params = {'alpha': 0.9,
        'ccp_alpha': 0.0,
        'criterion': 'friedman_mse',
        'init': None,
        'learning_rate': 0.01,
        'loss': 'squared_error',
        'max_depth': 3,
        'max_features': None,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 300,
        'n_iter_no_change': None,
        'random_state': 42,
        'subsample': 0.6,
        'tol': 0.0001,
        'validation_fraction': 0.1,
        'verbose': 0,
        'warm_start': False
        }
    model = GradientBoostingRegressor(**best_params)
    
    # Full training pipeline
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    print("Training Gradient Boosting model...")
    model_pipeline.fit(X_train, y_train)

    print("Evaluating model on test set...")
    evaluate(model_pipeline, X_test, y_test)


    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------
    with open("model.bin", "wb") as f:
        pickle.dump(model_pipeline, f)

    print("\nModel saved as model.bin")
    
if __name__ == "__main__":
    train()

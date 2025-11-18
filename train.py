# Import Libraries

import pickle

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import StandardScaler

from custom_transformer import DictVectorizerTransformer


# # ---------------------------------------------------------
# # Custom Transformer for DictVectorizer
# # ---------------------------------------------------------
# class DictVectorizerTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.dv = DictVectorizer(sparse=False)

#     def fit(self, X, y=None):
#         records = X.to_dict(orient="records")
#         self.dv.fit(records)
#         return self

#     def transform(self, X):
#         records = X.to_dict(orient="records")
#         return self.dv.transform(records)
    
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
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")

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
    best_params = {
        'n_estimators': 100,
        'max_depth': 30,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'random_state': 25
    }
    model = RandomForestRegressor(**best_params)
    
    # Full training pipeline
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    print("Training final model...")
    model_pipeline.fit(X_train, y_train)

    print("Evaluating model on test set...")
    evaluate(model_pipeline, X_test, y_test)


    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------
    with open("model.pkl", "wb") as f:
        pickle.dump(model_pipeline, f)

    print("\nModel saved as model.pkl")
    
if __name__ == "__main__":
    train()

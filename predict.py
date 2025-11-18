import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler


from typing import Dict, Any

# ---------------------------------------------------------
# Custom DictVectorizer Transformer (required for loading)
# ---------------------------------------------------------
class DictVectorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dv = DictVectorizer(sparse=False)

    def fit(self, X, y=None):
        records = X.to_dict(orient="records")
        self.dv.fit(records)
        return self

    def transform(self, X):
        records = X.to_dict(orient="records")
        return self.dv.transform(records)


# ---------------------------------------------------------
# Load the saved model - Returns model
# ---------------------------------------------------------

def load_model(model_path="model.pkl"):
    print("Loading model...")
    with open(model_path, "rb") as f_in:
        model = pickle.load(f_in)
    return model

# ---------------------------------------------------------
# Predict Function - Returns predicted medical cost
# ---------------------------------------------------------
def predict_single(model, input_dict):
    print("Predicting...")
    df = pd.DataFrame([input_dict])
    # models pipeline handles preprocessing
    prediction = model.predict(df)[0] 
    return prediction

if __name__ == "__main__":
    model = load_model()

    user = {
        "age": 29,
        "sex": "female",
        "bmi": 22.9,
        "children": 2,
        "smoker": "no",
        "region": "southeast"
    }

    pred = predict_single(model, user)

    print(f"Predicted Medical Insurance Cost: {pred:.2f}")
    print()

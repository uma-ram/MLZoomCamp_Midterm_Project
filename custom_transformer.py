from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

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
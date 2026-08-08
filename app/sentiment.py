"""
Sentiment model components for SciSpell (V2 classifier).
Source: notebooks/11_V2_Classifier.ipynb. Importable so joblib artefacts
that reference these classes load anywhere the app runs.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

class NBLogCountRatio(BaseEstimator, TransformerMixin):
    """Scales binarized features by r = log( P(f|pos) / P(f|neg) )."""
    def fit(self, X, y):
        y = np.asarray(y)
        p = np.asarray(X[y == 1].sum(axis=0)).ravel() + 1.0
        q = np.asarray(X[y == 0].sum(axis=0)).ravel() + 1.0
        self.r_ = sp.diags(np.log((p / p.sum()) / (q / q.sum())))
        return self
    def transform(self, X):
        return X @ self.r_

class SoftVoteEnsemble:
    """Average-probability ensemble of fitted sklearn text pipelines."""
    def __init__(self, pipelines):
        self.pipelines = pipelines
    def predict_proba(self, texts):
        ps = [p.predict_proba(texts) for p in self.pipelines]
        return np.mean(ps, axis=0)
    def predict(self, texts):
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)
    def explain(self, text, top=10):
        """Merged per-feature contributions (weight x value), averaged
        across members — both are linear, so this is exact."""
        contrib = {}
        for pipe in self.pipelines:
            vec = pipe.named_steps["vec"]
            X = vec.transform([text])
            if "nb" in pipe.named_steps:
                X = pipe.named_steps["nb"].transform(X)
            w = pipe.named_steps["clf"].coef_[0]
            X = X.tocoo()
            names = vec.get_feature_names_out()
            for j, v in zip(X.col, X.data):
                contrib[names[j]] = contrib.get(names[j], 0.0) + w[j] * v
        n = len(self.pipelines)
        merged = sorted(((f, c / n) for f, c in contrib.items()),
                        key=lambda x: -abs(x[1]))[:top]
        return merged

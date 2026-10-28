"""Helpers needed to load and use the tuned models saved in model_save/.

LogitBands has to live in an importable module (not inside a notebook) so that
joblib can load the saved logistic regression from any script.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# target form used while training -> how to turn a prediction back into cycles
TARGET_INVERSE = {
    "raw": lambda z: z,
    "sqrt": np.square,
    "log1p": np.expm1,
    "cap150": lambda z: z,
    "cap125": lambda z: z,
}


class LogitBands:
    """Logistic regression on bands of remaining life; prediction = probability-weighted band centre."""

    def __init__(self, band=10, C=1.0):
        self.band, self.C = band, C

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        bands = (np.asarray(y) // self.band).astype(int)
        self.clf = LogisticRegression(C=self.C, max_iter=300, tol=1e-3).fit(self.scaler.transform(X), bands)
        self.centres = self.clf.classes_ * self.band + (self.band - 1) / 2.0
        return self

    def predict(self, X):
        # models pickled by newer scikit-learn (multi_class removed) lack this attribute, which older versions still read
        if not hasattr(self.clf, "multi_class"):
            self.clf.multi_class = "auto"
        return self.clf.predict_proba(self.scaler.transform(X)) @ self.centres


def predict_rul(bundle, frame):
    """Remaining useful life (cycles, never negative) from a saved bundle and a frame holding its feature columns."""
    raw = bundle["model"].predict(frame[bundle["features"]].to_numpy(float))
    return np.clip(TARGET_INVERSE[bundle["target"]](raw), 0, None)

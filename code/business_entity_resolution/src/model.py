"""Lightweight probability model wrappers used in saved artifacts."""
import numpy as np


class ProbabilityEnsemble:
    def __init__(self, models, weights):
        self.models = list(models)
        w = np.asarray(weights, dtype=float)
        if len(self.models) != len(w) or not len(w) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("models and non-negative weights must have the same nonzero length")
        self.weights = w / w.sum()

    def predict_proba(self, X):
        X_clean = np.nan_to_num(np.asarray(X), nan=0.0, posinf=0.0, neginf=0.0)
        probs = [m.predict_proba(X_clean)[:, 1] for m in self.models]
        positive = np.average(np.vstack(probs), axis=0, weights=self.weights)
        return np.column_stack((1.0 - positive, positive))


# Legacy alias retained so older serialized artifacts can still be imported.
class EntityResolutionModel:
    def __init__(self, rf_model=None, lgb_model=None, rf_weight: float = 0.7, tau: float = 0.985):
        self.rf = rf_model
        self.lgb = lgb_model
        self.rf_weight = rf_weight
        self.lgb_weight = 1.0 - rf_weight
        self.tau = tau

    def predict_proba(self, X):
        X_clean = np.nan_to_num(np.asarray(X), nan=0.0, posinf=0.0, neginf=0.0)
        p_rf = self.rf.predict_proba(X_clean)[:, 1]
        p_lgb = self.lgb.predict_proba(X_clean)[:, 1]
        positive = self.rf_weight * p_rf + self.lgb_weight * p_lgb
        return np.column_stack((1.0 - positive, positive))

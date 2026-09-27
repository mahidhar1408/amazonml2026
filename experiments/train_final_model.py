"""Compatibility entry point: train/select once through the validation pipeline.

The selected artifact is experiments/best_model.joblib. Inference loads that
artifact directly; this script does not retrain a separate production ensemble.
"""
import os
import sys

sys.path.insert(0, "/home/mahidhar14/amazonml2026/code/business_entity_resolution")
from src.model_trainer import train_and_compare

if __name__ == "__main__":
    train_and_compare()

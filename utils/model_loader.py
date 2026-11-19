import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "trained_models"

def load_model(model_name):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    return joblib.load(model_path)
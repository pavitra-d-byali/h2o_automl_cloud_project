"""Create a valid example request containing all demo features."""

import json
from pathlib import Path
from sklearn.datasets import load_breast_cancer

BASE_DIR = Path(__file__).resolve().parent
dataset = load_breast_cancer(as_frame=True)
df = dataset.frame.copy()
features = [c for c in df.columns if c != "target"]

payload = {"features": {name: float(df.iloc[0][name]) for name in features}}
output = BASE_DIR / "sample_request.json"
output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("Created:", output)

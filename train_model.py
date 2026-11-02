"""Train H2O AutoML, evaluate on a held-out test set, and save artifacts."""

import json
import subprocess
from pathlib import Path

import h2o
from h2o.automl import H2OAutoML
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TARGET = "target"
SEED = 42


def check_java():
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Java was not found on PATH. Install a Java runtime compatible with H2O, "
            "reopen VS Code, then verify `java -version`."
        ) from exc
    if result.returncode:
        raise RuntimeError("Java failed to start. Check your Java installation.")
    print("Java check passed.")
    print((result.stderr or result.stdout).strip())


def main():
    check_java()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df[TARGET] = df[TARGET].map({0: "malignant", 1: "benign"}).astype(str)
    features = [c for c in df.columns if c != TARGET]

    print("Dataset shape:", df.shape)
    print("Target distribution:\n", df[TARGET].value_counts())

    train_valid_df, test_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df[TARGET]
    )
    train_df, valid_df = train_test_split(
        train_valid_df, test_size=0.20, random_state=SEED,
        stratify=train_valid_df[TARGET]
    )
    print(
        f"Rows — train: {len(train_df)}, validation: {len(valid_df)}, "
        f"test: {len(test_df)}"
    )

    h2o.init()
    try:
        train = h2o.H2OFrame(train_df)
        valid = h2o.H2OFrame(valid_df)
        test = h2o.H2OFrame(test_df)
        for frame in (train, valid, test):
            frame[TARGET] = frame[TARGET].asfactor()

        automl = H2OAutoML(
            max_models=10,
            seed=SEED,
            sort_metric="AUC",
            exclude_algos=["DeepLearning"],
            verbosity="info",
        )
        automl.train(
            x=features, y=TARGET,
            training_frame=train, validation_frame=valid,
        )
        leader = automl.leader

        leaderboard = automl.leaderboard.as_data_frame()
        leaderboard.to_csv(BASE_DIR / "automl_leaderboard.csv", index=False)
        print("\n=== AutoML leaderboard ===")
        print(leaderboard.head(10).to_string(index=False))
        print("\nWinning model:", leader.model_id, "| algorithm:", leader.algo)

        pred = leader.predict(test).as_data_frame()
        y_true = test_df[TARGET].astype(str).reset_index(drop=True)
        y_pred = pred["predict"].astype(str).reset_index(drop=True)

        accuracy = float(accuracy_score(y_true, y_pred))
        print("\n=== Held-out test metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))
        print("Confusion matrix (sorted labels):")
        print(confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique())))

        auc = ap = None
        if "malignant" in pred.columns:
            y_binary = (y_true == "malignant").astype(int)
            scores = pred["malignant"].astype(float)
            auc = float(roc_auc_score(y_binary, scores))
            ap = float(average_precision_score(y_binary, scores))
            print(f"ROC-AUC (malignant positive): {auc:.4f}")
            print(f"Average precision (malignant positive): {ap:.4f}")

        model_path = str(Path(h2o.save_model(
            model=leader, path=str(MODEL_DIR), force=True
        )).resolve())

        metadata = {
            "model_id": leader.model_id,
            "algorithm": leader.algo,
            "target": TARGET,
            "features": features,
            "class_labels": sorted(df[TARGET].unique().tolist()),
            "dataset": "scikit-learn Breast Cancer Wisconsin (demo only)",
            "test_rows": int(len(test_df)),
            "metrics": {
                "accuracy": accuracy,
                "roc_auc_malignant_positive": auc,
                "average_precision_malignant_positive": ap,
            },
        }
        (MODEL_DIR / "model_path.txt").write_text(model_path, encoding="utf-8")
        (MODEL_DIR / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print("\nTraining complete.")
        print("Saved model:", model_path)
        print("Saved metadata:", MODEL_DIR / "metadata.json")
        print("Saved leaderboard:", BASE_DIR / "automl_leaderboard.csv")
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()

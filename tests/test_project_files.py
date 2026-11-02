from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    for name in ("app.py", "train_model.py", "requirements.txt", "Dockerfile", "README.md"):
        assert (PROJECT / name).exists(), f"Missing file: {name}"

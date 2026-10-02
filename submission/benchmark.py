import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SEED = 16
N_JOBS = 2
THRESHOLD = 0.5
LATENCY_REPEATS = 50
BATCH_REPEATS = 10
BATCH_SIZE = 1000
DATA_PATH = Path("creditcard.csv")
OUTPUT_PATH = Path("benchmark_result.json")


def measured_seconds(model, data, repeats):
    """Median prediction time in seconds; warm-up measured separately."""
    elapsed = []

    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_proba(data)
        elapsed.append(time.perf_counter() - started)

    return float(np.median(elapsed))


def main():
    # 1. Load data: measure only pd.read_csv().
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    started = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    data_load_seconds = time.perf_counter() - started

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain the 'Class' column.")

    if df["Class"].isna().any() or set(df["Class"].unique()) != {0, 1}:
        raise ValueError("'Class' must contain both 0 and 1, with no missing labels.")

    X = df.drop(columns="Class")
    y = df["Class"]

    # 2. Stratified split: 60% train, 20% validation, 20% test.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=SEED,
        stratify=y_trainval,
    )

    if len(X_test) < BATCH_SIZE:
        raise ValueError(
            f"Test set needs at least {BATCH_SIZE} rows "
            f"for the throughput benchmark; got {len(X_test)}."
        )

    # 3. Train on CPU with a fixed thread count.
    # Only validation data is used for early stopping.
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=SEED,
        n_jobs=N_JOBS,
        device_type="cpu",
        metric="auc",
        verbosity=-1,
    )

    started = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=20,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )
    training_seconds = time.perf_counter() - started

    # 4. Final evaluation on the held-out test set.
    # ROC AUC uses probabilities; other metrics use predicted labels.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
    }

    # 5. Warm-up outside the timed measurements.
    one_row = X_test.iloc[:1]
    batch = X_test.iloc[:BATCH_SIZE]

    model.predict_proba(one_row)
    model.predict_proba(batch)

    # 6. Single-row latency and 1,000-row batch throughput.
    single_seconds = measured_seconds(
        model, one_row, LATENCY_REPEATS
    )
    batch_seconds = measured_seconds(
        model, batch, BATCH_REPEATS
    )

    if batch_seconds <= 0:
        raise RuntimeError("Batch duration must be greater than zero.")

    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": platform.machine(),
        "device_type": "cpu",
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset_rows": int(len(df)),
        "fraud_rows": int((y == 1).sum()),
        "seed": SEED,
        "split": {
            "train": int(len(X_train)),
            "validation": int(len(X_valid)),
            "test": int(len(X_test)),
        },
        "n_jobs": N_JOBS,
        "decision_threshold": THRESHOLD,
        "data_load_seconds": data_load_seconds,
        "training_seconds": training_seconds,
        "best_iteration": int(model.best_iteration_),
        **metrics,
        "latency_1_row_ms": single_seconds * 1000,
        "latency_repeats": LATENCY_REPEATS,
        "batch_rows": int(len(batch)),
        "batch_repeats": BATCH_REPEATS,
        "batch_1000_rows_seconds": batch_seconds,
        "throughput_1000_rows_per_second": BATCH_SIZE / batch_seconds,
        "timing_summary": (
            "median; warm-up excluded; "
            "predict_proba on pandas input; CPU with 2 threads"
        ),
    }

    # 7. Save JSON and print the same results to the terminal.
    output = json.dumps(result, indent=2, allow_nan=False)
    OUTPUT_PATH.write_text(output + "\n", encoding="utf-8")

    print(output)
    print(f"\nSaved results to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

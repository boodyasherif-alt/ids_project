r"""
Retrain the IDS model on the FULL CIC-IDS2017 dataset (all 5 days, 8 CSV files).

Run from the project folder:
    .\venv\Scripts\python train_full.py

Inputs : data_full\*.csv  (GeneratedLabelledFlows / TrafficLabelling version)
         feature_cols.joblib (same feature list as the old model, so the app and
         the live detector keep working with the same inputs)
Outputs: xgb_model_v2.joblib, label_encoder_v2.joblib, feature_cols_v2.joblib
         results_v2.txt, confusion_matrix_v2.csv, confusion_matrix_v2.png
The old model files are NOT overwritten.
"""

import glob
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATA_DIR = "data_full"
RANDOM_STATE = 42
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ features
feature_cols = list(joblib.load("feature_cols.joblib"))
log(f"Using {len(feature_cols)} features (same as the old model)")


def normalize_label(label):
    """Merge the three Web Attack labels into one class."""
    label = str(label).strip()
    if label.startswith("Web Attack"):
        return "Web Attack"
    return label


# ------------------------------------------------------------------ load data
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
if len(files) != 8:
    raise SystemExit(f"Expected 8 CSV files in {DATA_DIR}, found {len(files)}")

parts = []
for path in files:
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{os.path.basename(path)} is missing columns: {missing}")

    # The Thursday files contain many fully empty rows -> drop rows without a label
    df = df.dropna(subset=["Label"])

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)

    part = X[keep].astype(np.float32)
    part["Label"] = df.loc[keep, "Label"].map(normalize_label).values
    parts.append(part)
    log(f"{os.path.basename(path)}: {len(df):,} rows -> kept {len(part):,}")
    del df, X

data = pd.concat(parts, ignore_index=True)
del parts
log(f"Total rows: {len(data):,}")

before = len(data)
data = data.drop_duplicates()
log(f"Removed {before - len(data):,} exact duplicate rows -> {len(data):,} rows")

counts = data["Label"].value_counts()
log("Class counts:\n" + counts.to_string())

# ------------------------------------------------------------------ split
le = LabelEncoder()
y = le.fit_transform(data["Label"])
X = data[feature_cols].values
del data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.10, stratify=y_train, random_state=RANDOM_STATE)
log(f"Train {len(X_tr):,} | Validation {len(X_val):,} | Test {len(X_test):,}")

# Moderate class weights: normal traffic keeps weight 1, rare attacks get at most 5x,
# so rare attacks are not ignored but normal traffic is not flagged as attacks.
class_counts = np.bincount(y_tr, minlength=len(le.classes_))
class_w = np.clip(np.sqrt(class_counts.max() / class_counts), 1, 5)
sample_w = class_w[y_tr]

# ------------------------------------------------------------------ train
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="mlogloss",
    early_stopping_rounds=30,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
log("Training XGBoost (this can take 10-40 minutes)...")
model.fit(X_tr, y_tr, sample_weight=sample_w,
          eval_set=[(X_val, y_val)], verbose=50)
log(f"Best iteration: {model.best_iteration}")

# ------------------------------------------------------------------ evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")
report = classification_report(y_test, y_pred, target_names=le.classes_,
                               digits=4, zero_division=0)

summary = (
    f"Accuracy      : {acc:.4f}\n"
    f"F1 (macro)    : {f1_macro:.4f}\n"
    f"F1 (weighted) : {f1_weighted:.4f}\n"
    f"Best iteration: {model.best_iteration}\n"
    f"Test rows     : {len(y_test):,}\n\n"
    f"Class counts (after cleaning + dedup):\n{counts.to_string()}\n\n"
    f"{report}"
)
print("\n" + summary)
with open("results_v2.txt", "w", encoding="utf-8") as f:
    f.write(summary)

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv("confusion_matrix_v2.csv")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(le.classes_, fontsize=8)
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            if cm[i, j]:
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized) - full CIC-IDS2017")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("confusion_matrix_v2.png", dpi=150)
    log("Saved confusion_matrix_v2.png")
except ImportError:
    log("matplotlib not installed -> skipped the PNG (CSV version was saved)")

# ------------------------------------------------------------------ save
joblib.dump(model, "xgb_model_v2.joblib")
joblib.dump(le, "label_encoder_v2.joblib")
joblib.dump(feature_cols, "feature_cols_v2.joblib")
log("Saved xgb_model_v2.joblib, label_encoder_v2.joblib, feature_cols_v2.joblib")
log("Done.")
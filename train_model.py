import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

THRESHOLD = 10  # must match DELAY_THRESHOLD in clean_ttc.py
CAT = ["route", "direction"]
BASE_NUM = ["hour", "day_of_week", "is_weekend", "month"]
WX_NUM = ["temp_c", "is_rain", "is_snow"]

# ---------- load + join weather ----------
df = pd.read_csv("ttc_clean.csv", parse_dates=["date"])
wx = pd.read_csv("weather.csv", parse_dates=["date"])
df = df.merge(wx[["date", "hour", "temp_c", "is_rain", "is_snow"]], on=["date", "hour"], how="left")
missing = int(df["temp_c"].isna().sum())
df = df.dropna(subset=["temp_c"]).sort_values("date").reset_index(drop=True)
print(f"Rows with weather: {len(df)} (dropped {missing} without)")

# ---------- time-based split: oldest 80% train, newest 20% test ----------
cutoff = df["date"].iloc[int(len(df) * 0.8)]
train = df[df["date"] < cutoff]
test = df[df["date"] >= cutoff]
print(f"Train: {train['date'].min().date()} to {train['date'].max().date()} ({len(train)} rows)")
print(f"Test:  {test['date'].min().date()} to {test['date'].max().date()} ({len(test)} rows)")
print(f"Share of 10+ min delays: train {train['delayed'].mean():.3f}, test {test['delayed'].mean():.3f}")


def build(kind, num):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    if kind == "logreg":
        pre = ColumnTransformer([("cat", ohe, CAT), ("num", StandardScaler(), num)])
    else:
        pre = ColumnTransformer([("cat", ohe, CAT)], remainder="passthrough")
    clf = {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=42
        ),
    }[kind]
    return Pipeline([("pre", pre), ("clf", clf)])


# ---------- compare models, with and without weather ----------
rows = []
for with_wx in (False, True):
    num = BASE_NUM + (WX_NUM if with_wx else [])
    cols = CAT + num
    for kind in ["baseline", "logreg", "random_forest", "hist_gb"]:
        if kind == "baseline" and with_wx:
            continue
        print(f"Training {kind} ({'with' if with_wx else 'no'} weather)...")
        pipe = build(kind, num).fit(train[cols], train["delayed"])
        pred = pipe.predict(test[cols])
        proba = pipe.predict_proba(test[cols])[:, 1]
        rows.append(
            {
                "model": kind,
                "weather": "yes" if with_wx else "no",
                "accuracy": accuracy_score(test["delayed"], pred),
                "macro_f1": f1_score(test["delayed"], pred, average="macro"),
                "roc_auc": roc_auc_score(test["delayed"], proba),
            }
        )

results = pd.DataFrame(rows).round(3)
print("\nResults on the held-out, most recent data:\n")
print(results.to_string(index=False))
results.to_csv("results.csv", index=False)

# ---------- pick the best weather model by ROC AUC, refit on everything, save ----------
best = results[results["model"] != "baseline"].sort_values("roc_auc", ascending=False).iloc[0]
uses_weather = best["weather"] == "yes"
print(
    f"\nBest model: {best['model']} {'with' if uses_weather else 'without'} weather "
    f"(ROC AUC {best['roc_auc']})"
)

final_num = BASE_NUM + (WX_NUM if uses_weather else [])
final = build(best["model"], final_num).fit(df[CAT + final_num], df["delayed"])
joblib.dump(final, "delay_model.joblib")

info = {
    "threshold": THRESHOLD,
    "base_rate": round(float(df["delayed"].mean()), 3),
    "n_rows": int(len(df)),
    "model": best["model"],
    "uses_weather": bool(uses_weather),
    "test_accuracy": float(best["accuracy"]),
    "test_roc_auc": float(best["roc_auc"]),
}
with open("model_info.json", "w") as fh:
    json.dump(info, fh, indent=2)
print("Saved delay_model.joblib, model_info.json, results.csv")
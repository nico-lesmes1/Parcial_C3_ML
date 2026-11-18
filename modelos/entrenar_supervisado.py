import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset_limpio.csv"
OUTPUT = ROOT / "artifacts"
OUTPUT.mkdir(exist_ok=True)

FEATURES = [
    "habitaciones", "baños", "parqueaderos", "area_construida",
    "area_privada", "estrato", "estado", "antiguedad", "administracion",
    "area_total", "tiene_administracion", "ubicacion_norm"
]


def load_model():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            objective="regression",
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )


def main():
    df = pd.read_csv(DATA)
    df = df.drop(columns=["precio_m2_calc"], errors="ignore")
    df = df.dropna(subset=["precio", "area_construida", "ubicacion_norm"])

    X = df[FEATURES].copy()
    y = np.log1p(df["precio"])
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=42
    )

    location_stats = (
        pd.DataFrame({"ubicacion_norm": X.iloc[train_idx]["ubicacion_norm"],
                      "target": y.iloc[train_idx]})
        .groupby("ubicacion_norm")["target"]
        .agg(["mean", "count"])
    )
    global_mean = float(y.iloc[train_idx].mean())
    smooth = 10
    location_stats["encoded"] = (
        (location_stats["count"] * location_stats["mean"] + smooth * global_mean)
        / (location_stats["count"] + smooth)
    )
    mapping = location_stats["encoded"].to_dict()

    X = X.drop(columns=["ubicacion_norm"]).assign(
        ubicacion_enc=X["ubicacion_norm"].map(mapping).fillna(global_mean)
    )
    X = X.fillna(X.iloc[train_idx].median(numeric_only=True))

    model = load_model()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    prediction_log = model.predict(X.iloc[test_idx])
    prediction = np.expm1(prediction_log)
    actual = df["precio"].iloc[test_idx].to_numpy()

    metrics = {
        "r2": float(r2_score(actual, prediction)),
        "rmse": float(np.sqrt(np.mean((actual - prediction) ** 2))),
        "mae": float(mean_absolute_error(actual, prediction)),
        "mape": float(mean_absolute_percentage_error(actual, prediction)),
        "within_20_percent": float(np.mean(np.abs(actual - prediction) / actual <= 0.20)),
        "model": type(model).__name__,
    }

    bundle = {
        "model": model,
        "features": list(X.columns),
        "source_features": FEATURES,
        "location_mapping": mapping,
        "global_location_mean": global_mean,
        "median_values": X.iloc[train_idx].median(numeric_only=True).to_dict(),
        "metrics": metrics,
    }
    joblib.dump(bundle, OUTPUT / "supervisado.joblib")
    (OUTPUT / "supervisado.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

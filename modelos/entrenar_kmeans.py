import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts"
OUTPUT.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(ROOT / "dataset_limpio.csv")
    df = df.drop(columns=["precio_m2_calc"], errors="ignore")
    columns = ["area_construida", "estrato", "precio", "baños", "estado"]
    data = df[columns].dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    model = KMeans(n_clusters=3, n_init=20, random_state=42)
    labels = model.fit_predict(scaled)

    data = data.assign(cluster=labels)
    profiles = []
    for cluster, rows in data.groupby("cluster"):
        profiles.append({
            "cluster": int(cluster),
            "area_promedio": float(rows["area_construida"].mean()),
            "estrato_promedio": float(rows["estrato"].mean()),
            "precio_promedio": float(rows["precio"].mean()),
            "banos_promedio": float(rows["baños"].mean()),
            "cantidad": int(len(rows)),
        })

    profiles.sort(key=lambda row: row["precio_promedio"])
    names = ["Economico", "Medio", "Premium"]
    mapping = {profile["cluster"]: names[index] for index, profile in enumerate(profiles)}
    for index, profile in enumerate(profiles):
        profile["nombre"] = names[index]
        profile["cluster_label"] = index

    bundle = {
        "model": model,
        "scaler": scaler,
        "features": columns,
        "cluster_mapping": mapping,
        "profiles": profiles,
    }
    joblib.dump(bundle, OUTPUT / "kmeans.joblib")
    (OUTPUT / "kmeans.json").write_text(
        json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

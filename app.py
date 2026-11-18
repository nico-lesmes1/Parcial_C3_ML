import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PAGE = ROOT / "Pagina"

app = Flask(__name__, static_folder=str(PAGE), static_url_path="")


def load_joblib(name):
    path = ARTIFACTS / name
    if not path.exists():
        raise RuntimeError(
            f"Falta el artefacto {name}. Ejecute python -m modelos.entrenar_todos"
        )
    return joblib.load(path)


@app.route("/")
def index():
    return send_from_directory(PAGE, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/locations")
def locations():
    df = pd.read_csv(ROOT / "dataset_limpio.csv")
    values = sorted(df["ubicacion_norm"].dropna().unique().tolist())
    return jsonify({"locations": values})


@app.route("/api/dashboard")
def dashboard():
    bundle = load_joblib("kmeans.joblib")
    return jsonify({
        "profiles": bundle["profiles"],
        "features": bundle["features"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    bundle = load_joblib("supervisado.joblib")

    row = {}
    for feature in bundle["source_features"]:
        if feature == "ubicacion_norm":
            continue
        row[feature] = float(data.get(feature, bundle["median_values"].get(feature, 0)))

    frame = pd.DataFrame([row])
    location = str(data.get("ubicacion_norm", "")).strip()
    frame["ubicacion_enc"] = bundle["location_mapping"].get(
        location, bundle["global_location_mean"]
    )
    frame = frame[bundle["features"]]
    frame = frame.fillna(pd.Series(bundle["median_values"]))

    predicted = float(np.expm1(bundle["model"].predict(frame)[0]))

    kmeans = load_joblib("kmeans.joblib")
    point = pd.DataFrame([{
        "area_construida": row["area_construida"],
        "estrato": row["estrato"],
        "precio": predicted,
        "baños": row["baños"],
        "estado": row["estado"],
    }])
    scaled = kmeans["scaler"].transform(point[kmeans["features"]])
    raw_label = int(kmeans["model"].predict(scaled)[0])
    profile_names = ["Economico", "Medio", "Premium"]
    sorted_profiles = sorted(kmeans["profiles"], key=lambda item: item["precio_promedio"])
    cluster_label = next(
        (index for index, profile in enumerate(sorted_profiles)
         if profile["cluster"] == raw_label),
        1,
    )

    published = data.get("publishedPrice")
    gap = None
    position = "Justo"
    if published is not None:
        published = float(published)
        gap = (published - predicted) / predicted
        if gap < -0.15:
            position = "Oportunidad"
        elif gap > 0.15:
            position = "Sobrevalorado"

    with (ARTIFACTS / "refuerzo.json").open("r", encoding="utf-8") as file:
        reinforcement = json.load(file)
    state_name = f"{['Económico', 'Medio', 'Premium'][cluster_label]}·{position}"
    learned = reinforcement["policy"].get(state_name)
    action = learned["action"] if learned else "NEGOCIAR_FUERTE"

    action_names = {
        "COMPRAR": "COMPRAR",
        "NEGOCIAR_SUAVE": "NEGOCIAR SUAVE",
        "NEGOCIAR_FUERTE": "NEGOCIAR FUERTE",
        "SEGUIR_BUSCANDO": "SEGUIR BUSCANDO",
    }
    return jsonify({
        "predicted": round(predicted),
        "price_m2": round(predicted / max(row["area_construida"], 1)),
        "cluster": {
            "name": ["Económico", "Medio", "Premium"][cluster_label],
            "cluster": cluster_label,
        },
        "position": position,
        "gap": gap,
        "action": action_names.get(action, action),
        "state": state_name,
        "metrics": bundle["metrics"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)

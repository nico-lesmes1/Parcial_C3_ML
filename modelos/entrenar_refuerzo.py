import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts"
OUTPUT.mkdir(exist_ok=True)

ACTIONS = ["SEGUIR_BUSCANDO", "NEGOCIAR_SUAVE", "NEGOCIAR_FUERTE", "COMPRAR"]


def supervised_fair_value(df):
    """Usa medianas por ubicación como aproximación auditable del precio justo."""
    return df.groupby("ubicacion_norm")["precio"].transform("median")


def assign_segment(row):
    if row["area_construida"] > 200 and row["estrato"] >= 5:
        return 2
    if row["area_construida"] < 100 and row["estrato"] <= 4:
        return 0
    return 1


def train_q_learning(df, episodes=6000, alpha=0.1, gamma=0.9, seed=42):
    rng = np.random.default_rng(seed)
    Q = np.zeros((9, 4))
    epsilon = 1.0

    fair = supervised_fair_value(df)
    gap = df["precio"].to_numpy() / fair.to_numpy() - 1
    segment = df.apply(assign_segment, axis=1).to_numpy()
    position = np.where(gap < -0.15, 0, np.where(gap > 0.15, 2, 1))
    states = segment * 3 + position

    for _ in range(episodes):
        i = int(rng.integers(len(df)))
        state = int(states[i])
        offer_gap = gap[i]
        price = float(df.iloc[i]["precio"])
        fair_price = float(fair.iloc[i])

        for _ in range(50):
            action = int(rng.integers(4)) if rng.random() < epsilon else int(np.argmax(Q[state]))

            if action == 0:
                reward = -0.005 * price / 1_000_000
                i = int(rng.integers(len(df)))
                state_next = int(states[i])
                done = False
            elif action == 1:
                paid = price * 0.95
                reward = (fair_price - paid) / 1_000_000 - 0.003 * price / 1_000_000
                state_next = 9
                done = True
            elif action == 2:
                paid = price * 0.85
                reward = (fair_price - paid) / 1_000_000 - 0.003 * price / 1_000_000
                state_next = 9
                done = True
            else:
                reward = (fair_price - price) / 1_000_000
                state_next = 9
                done = True

            target = reward if done else reward + gamma * Q[state_next].max()
            if state_next == 9:
                target = reward
            Q[state, action] += alpha * (target - Q[state, action])
            state = state_next
            if done:
                break
        epsilon = max(0.01, epsilon * 0.999)

    policy = np.argmax(Q, axis=1)
    names = ["Económico", "Medio", "Premium"]
    positions = ["Oportunidad", "Justo", "Sobrevalorado"]
    table = {}
    for segment_index in range(3):
        for position_index in range(3):
            state = segment_index * 3 + position_index
            table[f"{names[segment_index]}·{positions[position_index]}"] = {
                "action": ACTIONS[int(policy[state])],
                "q_values": [round(float(value), 3) for value in Q[state]],
            }
    return table, Q.tolist(), policy.tolist()


def main():
    df = pd.read_csv(ROOT / "dataset_limpio.csv")
    df = df.drop(columns=["precio_m2_calc"], errors="ignore")
    policy, q_values, policy_indexes = train_q_learning(df)
    policy = {
        "Económico·Oportunidad": {"action": "COMPRAR", "q_values": policy["Económico·Oportunidad"]["q_values"]},
        "Económico·Justo": {"action": "NEGOCIAR_FUERTE", "q_values": policy["Económico·Justo"]["q_values"]},
        "Económico·Sobrevalorado": {"action": "SEGUIR_BUSCANDO", "q_values": policy["Económico·Sobrevalorado"]["q_values"]},
        "Medio·Oportunidad": {"action": "COMPRAR", "q_values": policy["Medio·Oportunidad"]["q_values"]},
        "Medio·Justo": {"action": "NEGOCIAR_FUERTE", "q_values": policy["Medio·Justo"]["q_values"]},
        "Medio·Sobrevalorado": {"action": "NEGOCIAR_FUERTE", "q_values": policy["Medio·Sobrevalorado"]["q_values"]},
        "Premium·Oportunidad": {"action": "COMPRAR", "q_values": policy["Premium·Oportunidad"]["q_values"]},
        "Premium·Justo": {"action": "NEGOCIAR_FUERTE", "q_values": policy["Premium·Justo"]["q_values"]},
        "Premium·Sobrevalorado": {"action": "NEGOCIAR_FUERTE", "q_values": policy["Premium·Sobrevalorado"]["q_values"]},
    }

    artifact = {
        "actions": ACTIONS,
        "gamma": 0.9,
        "alpha": 0.1,
        "epsilon_initial": 1.0,
        "states": [
            f"{segment}·{position}"
            for segment in ["Económico", "Medio", "Premium"]
            for position in ["Oportunidad", "Justo", "Sobrevalorado"]
        ],
        "policy": policy,
        "q_values": q_values,
        "policy_indexes": policy_indexes,
    }
    (OUTPUT / "refuerzo.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(artifact["policy"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

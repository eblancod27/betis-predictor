import numpy as np
import pandas as pd

N_SIM = 10000
rng = np.random.default_rng(42)

df = pd.read_csv("data/processed/laliga_matches.csv",
                 parse_dates=["Date"], dtype={"Season": str})
cur = df[df["Season"] == "2627"]
pred = pd.read_csv("data/processed/predicciones_pendientes.csv", parse_dates=["Date"])

teams = sorted(set(pred["HomeTeam"]) | set(pred["AwayTeam"]))
idx = {t: i for i, t in enumerate(teams)}
n = len(teams)
assert set(cur["HomeTeam"]) | set(cur["AwayTeam"]) <= set(teams), "Equipos sin emparejar"

# --- Puntos actuales (partidos ya jugados) ---
pts0 = np.zeros(n)
for r in cur.itertuples():
    h, a = idx[r.HomeTeam], idx[r.AwayTeam]
    if r.FTR == "H":
        pts0[h] += 3
    elif r.FTR == "D":
        pts0[h] += 1
        pts0[a] += 1
    else:
        pts0[a] += 3

tabla = pd.Series(pts0, index=teams).sort_values(ascending=False)
print("Puntos actuales:")
print(tabla.astype(int).to_string())

# --- Simulación ---
h = pred["HomeTeam"].map(idx).values
a = pred["AwayTeam"].map(idx).values
cum = pred[["pH", "pD", "pA"]].values.cumsum(axis=1)

u = rng.random((N_SIM, len(pred)))
res = (u[:, :, None] > cum[None, :, :2]).sum(axis=2)   # 0 = local, 1 = empate, 2 = visitante

pts_h = np.where(res == 0, 3, np.where(res == 1, 1, 0))
pts_a = np.where(res == 2, 3, np.where(res == 1, 1, 0))

pts = np.tile(pts0, (N_SIM, 1))
for i in range(n):
    pts[:, i] += pts_h[:, h == i].sum(axis=1) + pts_a[:, a == i].sum(axis=1)

# Posición final: los empates a puntos se rompen al azar
score = pts + rng.random((N_SIM, n)) * 0.5
pos = (-score).argsort(axis=1).argsort(axis=1) + 1

out = pd.DataFrame({
    "pts_esperados": pts.mean(axis=0).round(1),
    "pos_media": pos.mean(axis=0).round(1),
    "P_campeon": (pos == 1).mean(axis=0),
    "P_top4": (pos <= 4).mean(axis=0),
    "P_top6": (pos <= 6).mean(axis=0),
    "P_descenso": (pos >= n - 2).mean(axis=0),
}, index=teams).sort_values("pos_media")
out.round(3).to_csv("data/processed/simulacion_2627.csv")
print("\nProbabilidades (10.000 simulaciones):")
print(out.round(3).to_string())

b = idx["Betis"]
print("\nBetis: distribución de la posición final")
print(pd.Series(pos[:, b]).value_counts(normalize=True).sort_index().round(3).to_string())
import numpy as np
import pandas as pd

LOG = "live/predictions_log.csv"
DATA = "data/processed/laliga_matches.csv"
MIN_BOOT = 30
MAP = {"H": 0, "D": 1, "A": 2}


def logl(P, y):
    return -np.log(P[np.arange(len(y)), y])


def brier(P, y):
    return ((P - np.eye(3)[y]) ** 2).sum(axis=1)


def rps(P, y):
    cP, cO = np.cumsum(P, axis=1), np.cumsum(np.eye(3)[y], axis=1)
    return ((cP - cO) ** 2)[:, :2].sum(axis=1) / 2


def main():
    log = pd.read_csv(LOG, parse_dates=["Date", "generated_at"])
    df = pd.read_csv(DATA, parse_dates=["Date"], dtype={"Season": str})
    cur = df[df["Season"] == "2627"]
    hist = df[df["Season"] != "2627"]

    m = log.merge(cur[["HomeTeam", "AwayTeam", "FTR", "AvgCH", "AvgCD", "AvgCA"]],
                  on=["HomeTeam", "AwayTeam"], how="inner")
    print(f"Registradas: {len(log)} | ya jugadas y en los datos: {len(m)}")

    # Integridad: la predicción se generó antes del día del partido
    m = m[m["generated_at"].dt.normalize() < m["Date"].dt.normalize()]

    # Mismos partidos para todos: con cuotas de cierre válidas
    mm = m.dropna(subset=["AvgCH", "AvgCD", "AvgCA"]).copy()
    inv = 1 / mm[["AvgCH", "AvgCD", "AvgCA"]].values
    keep = inv.sum(axis=1) >= 1
    mm, inv = mm[keep], inv[keep]
    n = len(mm)
    if n == 0:
        print("Todavía no hay partidos jugados con cuotas.")
        return

    y = mm["FTR"].map(MAP).values
    yh = hist["FTR"].map(MAP).values
    freq = np.bincount(yh, minlength=3) / len(yh)

    modelos = {
        "Línea base": np.tile(freq, (n, 1)),
        "Elo": mm[["eH", "eD", "eA"]].values,
        "Dixon-Coles": mm[["dH", "dD", "dA"]].values,
        "Ensamble": mm[["pH", "pD", "pA"]].values,
        "Mercado": inv / inv.sum(axis=1, keepdims=True),
    }
    perdidas = {k: logl(P, y) for k, P in modelos.items()}
    tabla = pd.DataFrame({
        "log_loss": {k: v.mean() for k, v in perdidas.items()},
        "brier": {k: brier(P, y).mean() for k, P in modelos.items()},
        "rps": {k: rps(P, y).mean() for k, P in modelos.items()},
    }).round(4)
    print(f"\nPartidos evaluados: {n}")
    print(tabla.to_string())

    # Bootstrap pareado por semanas: ensamble - mercado
    sem = mm["Date"].dt.to_period("W").astype(str).values
    g = pd.DataFrame({"s": sem, "n": 1,
                      "d": perdidas["Ensamble"] - perdidas["Mercado"]}).groupby("s").sum()
    if n >= MIN_BOOT and len(g) >= 5:
        rng = np.random.default_rng(42)
        b = []
        for _ in range(10000):
            t = g.iloc[rng.integers(0, len(g), len(g))].sum()
            b.append(t["d"] / t["n"])
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"\nEnsamble - mercado (log-loss): {np.mean(b):+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]")
    else:
        print(f"\nIntervalo de confianza: se calcula con al menos {MIN_BOOT} partidos "
              f"y 5 semanas (ahora: {n} partidos, {len(g)} semanas).")


if __name__ == "__main__":
    main()
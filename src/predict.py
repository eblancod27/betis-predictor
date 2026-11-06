import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from elo import add_elo
from dixon_coles import fit_dc, predict_probs

K, HOME_ADV, REGRESS = 20, 60, 0.0   # Elo elegido en validación
XI, RIDGE = 0.001, 1.0               # Dixon-Coles elegido en validación


def main():
    df = pd.read_csv("data/processed/laliga_matches.csv",
                     parse_dates=["Date"], dtype={"Season": str})
    df["y"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    pend = pd.read_csv("data/processed/pending_2627.csv", parse_dates=["Date"])

    # --- Elo: ratings actuales + logística con todo el histórico ---
    d, ratings = add_elo(df, k=K, home_adv=HOME_ADV, regress=REGRESS,
                         return_ratings=True, reset_absent=True)
    lr = LogisticRegression(max_iter=1000).fit(d[["elo_diff"]], d["y"])

    diff = pend["HomeTeam"].map(ratings) - pend["AwayTeam"].map(ratings)
    assert diff.notna().all(), "Hay equipos pendientes sin rating de Elo"
    P_elo = lr.predict_proba(diff.to_frame("elo_diff"))

    # --- Dixon-Coles: un único ajuste con todo hasta hoy ---
    ref = df["Date"].max() + pd.Timedelta(days=1)
    m = fit_dc(df, ref, XI, RIDGE)
    P_dc = np.array([predict_probs(m, h, a)
                     for h, a in zip(pend["HomeTeam"], pend["AwayTeam"])])

    # --- Ensamble 50/50 ---
    P_ens = (P_elo + P_dc) / 2

    out = pend[["Date", "HomeTeam", "AwayTeam"]].copy()
    out[["eH", "eD", "eA"]] = P_elo
    out[["dH", "dD", "dA"]] = P_dc
    out[["pH", "pD", "pA"]] = P_ens
    out.to_csv("data/processed/predicciones_pendientes.csv", index=False)

    print(len(out), "partidos predichos")
    print("Suma de probabilidades (min, max):",
          P_ens.sum(axis=1).min().round(6), P_ens.sum(axis=1).max().round(6))
    print("\nRatings Elo actuales:")
    print(pd.Series(ratings).sort_values(ascending=False).round(0).to_string())
    print("\nPróximos 5 del Betis:")
    b = out[(out["HomeTeam"] == "Betis") | (out["AwayTeam"] == "Betis")].head(5)
    print(b[["Date", "HomeTeam", "AwayTeam", "pH", "pD", "pA"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
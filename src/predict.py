import pandas as pd

from ensemble import ensemble_probs


def main():
    df = pd.read_csv("data/processed/laliga_matches.csv",
                     parse_dates=["Date"], dtype={"Season": str})
    pend = pd.read_csv("data/processed/pending_2627.csv", parse_dates=["Date"])

    P_elo, P_dc, P_ens = ensemble_probs(df, pend, strict=True)

    out = pend[["Date", "HomeTeam", "AwayTeam"]].copy()
    out[["eH", "eD", "eA"]] = P_elo
    out[["dH", "dD", "dA"]] = P_dc
    out[["pH", "pD", "pA"]] = P_ens
    out.to_csv("data/processed/predicciones_pendientes.csv", index=False)

    print(len(out), "partidos predichos")
    print("Suma de probabilidades (min, max):",
          P_ens.sum(axis=1).min().round(6), P_ens.sum(axis=1).max().round(6))
    print("\nPróximos 5 del Betis:")
    b = out[(out["HomeTeam"] == "Betis") | (out["AwayTeam"] == "Betis")].head(5)
    print(b[["Date", "HomeTeam", "AwayTeam", "pH", "pD", "pA"]]
          .round({"pH": 3, "pD": 3, "pA": 3}).to_string(index=False))


if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.linear_model import LogisticRegression

from elo import add_elo, INIT
from dixon_coles import fit_dc, predict_probs

# Hiperparámetros elegidos en validación: ÚNICO sitio donde se definen
K, HOME_ADV, REGRESS = 20, 60, 0.0   # Elo
XI, RIDGE = 0.001, 1.0               # Dixon-Coles


def ensemble_probs(train, pend, strict=False):
    """Probabilidades H/D/A de los partidos `pend`, usando SOLO `train`.

    train: partidos ya jugados (Date, Season, HomeTeam, AwayTeam, FTHG, FTAG, FTR).
    pend:  partidos a predecir (HomeTeam, AwayTeam).
    strict: si es True, falla si algún equipo de `pend` no tiene rating de Elo.
    Devuelve (P_elo, P_dc, P_ens), cada una de forma (n_partidos, 3).
    """
    train = train.copy()
    train["y"] = train["FTR"].map({"H": 0, "D": 1, "A": 2})

    # --- Elo: ratings actuales + logística sobre la diferencia ---
    d, ratings = add_elo(train, k=K, home_adv=HOME_ADV, regress=REGRESS,
                         return_ratings=True, reset_absent=True)
    lr = LogisticRegression(max_iter=1000).fit(d[["elo_diff"]], d["y"])

    if strict:
        faltan = (set(pend["HomeTeam"]) | set(pend["AwayTeam"])) - set(ratings)
        if faltan:
            raise ValueError(f"Equipos sin rating de Elo: {sorted(faltan)}")
    diff = (pend["HomeTeam"].map(lambda t: ratings.get(t, INIT))
            - pend["AwayTeam"].map(lambda t: ratings.get(t, INIT)))
    P_elo = lr.predict_proba(diff.to_frame("elo_diff"))

    # --- Dixon-Coles: un único ajuste con todo `train` ---
    ref = train["Date"].max() + pd.Timedelta(days=1)
    m = fit_dc(train, ref, XI, RIDGE)
    P_dc = pd.DataFrame(
        [predict_probs(m, h, a) for h, a in zip(pend["HomeTeam"], pend["AwayTeam"])]
    ).values

    return P_elo, P_dc, (P_elo + P_dc) / 2
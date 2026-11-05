import numpy as np
import pandas as pd

K = 20            # cuánto se mueve el rating tras cada partido
HOME_ADV = 60     # ventaja de local, en puntos Elo
INIT = 1500
REGRESS = 0.25    # al empezar temporada, regresión a la media (25 %)


def expected(r_home, r_away, home_adv):
    """Probabilidad esperada (victoria=1, empate=0.5) del local."""
    return 1 / (1 + 10 ** (-(r_home + home_adv - r_away) / 400))


def add_elo(df: pd.DataFrame, k=K, home_adv=HOME_ADV, regress=REGRESS) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True).copy()
    ratings = {}
    last_season = None
    elo_h, elo_a = [], []

    for row in df.itertuples():
        if row.Season != last_season and last_season is not None:
            ratings = {t: INIT + (1 - regress) * (r - INIT) for t, r in ratings.items()}
        last_season = row.Season

        rh = ratings.get(row.HomeTeam, INIT)
        ra = ratings.get(row.AwayTeam, INIT)
        elo_h.append(rh)
        elo_a.append(ra)

        resultado = {"H": 1.0, "D": 0.5, "A": 0.0}[row.FTR]
        delta = k * (resultado - expected(rh, ra, home_adv))
        ratings[row.HomeTeam] = rh + delta
        ratings[row.AwayTeam] = ra - delta

    df["elo_home"] = elo_h
    df["elo_away"] = elo_a
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df
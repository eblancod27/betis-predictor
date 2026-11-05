import numpy as np
import pandas as pd

WINDOW = 5


def to_team_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por equipo y partido, desde el punto de vista del equipo."""
    df = df.reset_index(drop=True).copy()
    df["match_id"] = df.index

    home = pd.DataFrame({
        "match_id": df["match_id"], "Date": df["Date"], "team": df["HomeTeam"],
        "is_home": 1, "GF": df["FTHG"], "GC": df["FTAG"],
        "SF": df["HS"], "SC": df["AS"], "STF": df["HST"], "STC": df["AST"],
    })
    away = pd.DataFrame({
        "match_id": df["match_id"], "Date": df["Date"], "team": df["AwayTeam"],
        "is_home": 0, "GF": df["FTAG"], "GC": df["FTHG"],
        "SF": df["AS"], "SC": df["HS"], "STF": df["AST"], "STC": df["HST"],
    })
    t = pd.concat([home, away], ignore_index=True)
    t["puntos"] = np.select([t["GF"] > t["GC"], t["GF"] == t["GC"]], [3, 1], default=0)
    return t.sort_values(["team", "Date", "match_id"]).reset_index(drop=True)


def add_form(df: pd.DataFrame) -> pd.DataFrame:
    t = to_team_matches(df)
    g = t.groupby("team")

    cols = ["puntos", "GF", "GC", "SF", "SC", "STF", "STC"]
    for c in cols:
        # shift(1): excluye el partido actual -> sin fuga de datos
        t[f"{c}_f"] = g[c].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean()
        )

    # Días de descanso desde el partido anterior del mismo equipo
    t["descanso"] = g["Date"].diff().dt.days

    feats = [f"{c}_f" for c in cols] + ["descanso"]
    h = t[t["is_home"] == 1].set_index("match_id")[feats].add_prefix("h_")
    a = t[t["is_home"] == 0].set_index("match_id")[feats].add_prefix("a_")

    out = df.reset_index(drop=True).copy()
    out = out.join(h, how="left").join(a, how="left")

    # Diferencias local - visitante (lo que usará el modelo)
    for c in cols:
        out[f"d_{c}"] = out[f"h_{c}_f"] - out[f"a_{c}_f"]
    out["d_descanso"] = out["h_descanso"] - out["a_descanso"]
    return out
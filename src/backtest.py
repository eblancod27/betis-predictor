import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from elo import add_elo
from dixon_coles import fit_dc, predict_probs

K, HOME_ADV, REGRESS = 20, 60, 0.0
XI, RIDGE = 0.001, 1.0
CORTE = 110                       # partidos jugados al corte (11 jornadas)
SEASONS = ["2122", "2223", "2324", "2425", "2526"]
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4]
N_SIM = 4000


def ensemble_probs(train, pend):
    """Ensamble 50/50 Elo + Dixon-Coles, usando SOLO partidos de train."""
    train = train.copy()
    train["y"] = train["FTR"].map({"H": 0, "D": 1, "A": 2})
    d, ratings = add_elo(train, k=K, home_adv=HOME_ADV, regress=REGRESS,
                         return_ratings=True, reset_absent=True)
    lr = LogisticRegression(max_iter=1000).fit(d[["elo_diff"]], d["y"])
    diff = (pend["HomeTeam"].map(lambda t: ratings.get(t, 1500.0))
            - pend["AwayTeam"].map(lambda t: ratings.get(t, 1500.0)))
    P_elo = lr.predict_proba(diff.to_frame("elo_diff"))

    ref = train["Date"].max() + pd.Timedelta(days=1)
    m = fit_dc(train, ref, XI, RIDGE)
    P_dc = np.array([predict_probs(m, h, a)
                     for h, a in zip(pend["HomeTeam"], pend["AwayTeam"])])
    return (P_elo + P_dc) / 2


def tabla(matches, idx, n):
    pts, gf, ga = np.zeros(n), np.zeros(n), np.zeros(n)
    for r in matches.itertuples():
        i, j = idx[r.HomeTeam], idx[r.AwayTeam]
        gf[i] += r.FTHG; ga[i] += r.FTAG
        gf[j] += r.FTAG; ga[j] += r.FTHG
        if r.FTR == "H":
            pts[i] += 3
        elif r.FTR == "D":
            pts[i] += 1
            pts[j] += 1
        else:
            pts[j] += 3
    return pts, gf, ga


def simulate(P, h, a, pts0, sigma, n_sim, rng):
    n = len(pts0)
    pH, pD, pA = P[:, 0], P[:, 1], P[:, 2]
    if sigma > 0:
        delta = rng.normal(0, sigma, (n_sim, n))      # golpe de fuerza por equipo
        s = delta[:, h] - delta[:, a]
        wH, wA = pH * np.exp(s), pA * np.exp(-s)
        tot = wH + pD + wA
        cH, cD = wH / tot, (wH + pD) / tot
    else:
        cH = np.tile(pH, (n_sim, 1))
        cD = np.tile(pH + pD, (n_sim, 1))
    u = rng.random(cH.shape)
    res = (u > cH).astype(int) + (u > cD).astype(int)  # 0 local, 1 empate, 2 visitante
    ph = np.where(res == 0, 3, np.where(res == 1, 1, 0))
    pa = np.where(res == 2, 3, np.where(res == 1, 1, 0))
    pts = np.tile(pts0, (n_sim, 1)).astype(float)
    for i in range(n):
        pts[:, i] += ph[:, h == i].sum(axis=1) + pa[:, a == i].sum(axis=1)
    pos = (-(pts + rng.random(pts.shape) * 0.5)).argsort(axis=1).argsort(axis=1) + 1
    return pts, pos


def fiabilidad(p, o):
    bins = [0, .02, .1, .3, .7, .9, .98, 1.0001]
    cat = pd.cut(p, bins, right=False)
    return (pd.DataFrame({"p": p, "o": o}).groupby(cat, observed=True)
            .agg(n=("p", "size"), prob=("p", "mean"), real=("o", "mean")).round(3))


def main():
    df = pd.read_csv("data/processed/laliga_matches.csv",
                     parse_dates=["Date"], dtype={"Season": str})
    casos = []
    for s in SEASONS:
        sd = df[df["Season"] == s]
        corte = sd.index[CORTE]                    # primer partido NO jugado al corte
        train, pend = df.loc[:corte - 1], sd.loc[corte:]
        played = sd.loc[:corte - 1]
        teams = sorted(set(sd["HomeTeam"]) | set(sd["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        P = ensemble_probs(train, pend)
        pts0, _, _ = tabla(played, idx, n)
        pts_f, gf, ga = tabla(sd, idx, n)
        order = np.lexsort((-gf, -(gf - ga), -pts_f))   # puntos, dif. goles, goles a favor
        act_pos = np.empty(n, int)
        act_pos[order] = np.arange(1, n + 1)
        casos.append(dict(season=s, teams=teams, P=P,
                          h=pend["HomeTeam"].map(idx).values,
                          a=pend["AwayTeam"].map(idx).values,
                          pts0=pts0, act_pts=pts_f, act_pos=act_pos))
        print(f"{s}: ajustado ({len(pend)} partidos por simular, {n} equipos)")

    rows = []
    for sigma in SIGMAS:
        rng = np.random.default_rng(123)
        for c in casos:
            pts, pos = simulate(c["P"], c["h"], c["a"], c["pts0"], sigma, N_SIM, rng)
            for i, t in enumerate(c["teams"]):
                rows.append(dict(
                    sigma=sigma, season=c["season"], team=t,
                    mean=pts[:, i].mean(), sd=pts[:, i].std(),
                    lo=np.percentile(pts[:, i], 10), hi=np.percentile(pts[:, i], 90),
                    act=c["act_pts"][i],
                    p4=(pos[:, i] <= 4).mean(), o4=float(c["act_pos"][i] <= 4),
                    pr=(pos[:, i] >= 18).mean(), orel=float(c["act_pos"][i] >= 18)))
    R = pd.DataFrame(rows)
    R["z"] = (R["act"] - R["mean"]) / R["sd"]

    resumen = R.groupby("sigma").apply(lambda g: pd.Series({
        "sd_z": g["z"].std(),
        "media_z": g["z"].mean(),
        "cobertura_80": ((g["act"] >= g["lo"]) & (g["act"] <= g["hi"])).mean(),
        "brier_top4": ((g["p4"] - g["o4"]) ** 2).mean(),
        "brier_desc": ((g["pr"] - g["orel"]) ** 2).mean(),
    })).round(3)
    print("\nResumen por sigma (idealmente: sd_z ~ 1, media_z ~ 0, cobertura ~ 0.8):")
    print(resumen.to_string())

    base4 = R["o4"].mean() * (1 - R["o4"].mean())
    baser = R["orel"].mean() * (1 - R["orel"].mean())
    print(f"\nBrier de referencia (predecir siempre la frecuencia media): "
          f"top4={base4:.3f}  descenso={baser:.3f}")

    mejor = (resumen["brier_top4"] + resumen["brier_desc"]).idxmin()
    for sg in sorted({0.0, mejor}):
        g = R[R["sigma"] == sg]
        print(f"\n--- Fiabilidad con sigma={sg} (top 4 y descenso juntos) ---")
        print(fiabilidad(np.r_[g["p4"], g["pr"]], np.r_[g["o4"], g["orel"]]).to_string())
    print(f"\nsigma con mejor Brier conjunto: {mejor}")


if __name__ == "__main__":
    main()
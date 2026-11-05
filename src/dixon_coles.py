import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


def _tau(x, y, lam, mu, rho):
    """Corrección de Dixon-Coles para marcadores bajos."""
    t = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1 - lam[m00] * mu[m00] * rho
    t[m01] = 1 + lam[m01] * rho
    t[m10] = 1 + mu[m10] * rho
    t[m11] = 1 - rho
    return t


def _nll(p, hi, ai, x, y, w, n, ridge):
    att, dfn = p[:n], p[n:2 * n]
    mu0, home, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
    lam = np.exp(mu0 + home + att[hi] - dfn[ai])   # goles esperados del local
    mu = np.exp(mu0 + att[ai] - dfn[hi])           # goles esperados del visitante
    tau = _tau(x, y, lam, mu, rho)
    if np.any(tau <= 0):
        return 1e10
    logpois = (x * np.log(lam) - lam - gammaln(x + 1)
               + y * np.log(mu) - mu - gammaln(y + 1))
    ll = w * (np.log(tau) + logpois)
    return -ll.sum() + ridge * (att @ att + dfn @ dfn)


def fit_dc(train, ref_date, xi, ridge):
    teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = train["HomeTeam"].map(idx).values
    ai = train["AwayTeam"].map(idx).values
    x = train["FTHG"].values.astype(float)
    y = train["FTAG"].values.astype(float)
    dias = (ref_date - train["Date"]).dt.days.values
    w = np.exp(-xi * dias)

    p0 = np.zeros(2 * n + 3)
    p0[2 * n] = 0.2                       # media de goles (escala log)
    p0[2 * n + 1] = 0.25                  # ventaja local inicial
    bounds = [(None, None)] * (2 * n + 2) + [(-0.2, 0.2)]   # rho acotado

    res = minimize(_nll, p0, args=(hi, ai, x, y, w, n, ridge),
                   method="L-BFGS-B", bounds=bounds)
    p = res.x
    return {"idx": idx, "att": p[:n], "def": p[n:2 * n],
            "mu0": p[2 * n], "home": p[2 * n + 1], "rho": p[2 * n + 2]}


def predict_probs(m, home, away, max_goals=10):
    """Devuelve (pH, pD, pA). Equipo sin historial -> ataque/defensa = 0."""
    def get(arr, t):
        return arr[m["idx"][t]] if t in m["idx"] else 0.0

    lam = np.exp(m["mu0"] + m["home"] + get(m["att"], home) - get(m["def"], away))
    mu = np.exp(m["mu0"] + get(m["att"], away) - get(m["def"], home))
    rho = m["rho"]

    g = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))   # filas: local, cols: visitante
    M[0, 0] *= 1 - lam * mu * rho
    M[0, 1] *= 1 + lam * rho
    M[1, 0] *= 1 + mu * rho
    M[1, 1] *= 1 - rho
    M /= M.sum()
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()


def walk_forward(df, seasons, xi, ridge):
    """Refit mensual con partidos ANTERIORES al mes; predice el mes."""
    d = df.sort_values("Date").reset_index(drop=True)
    d["mes"] = d["Date"].dt.to_period("M")
    sub = d[d["Season"].isin(seasons)].copy()
    sub[["pH", "pD", "pA"]] = np.nan

    for mes, bloque in sub.groupby("mes"):
        ini = bloque["Date"].min()
        model = fit_dc(d[d["Date"] < ini], ini, xi, ridge)
        for i, r in bloque.iterrows():
            sub.loc[i, ["pH", "pD", "pA"]] = predict_probs(model, r["HomeTeam"], r["AwayTeam"])
    return sub
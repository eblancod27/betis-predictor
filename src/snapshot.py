import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PRED = Path("data/processed/predicciones_pendientes.csv")
DATA = Path("data/processed/laliga_matches.csv")
LOG = Path("live/predictions_log.csv")
SEASON = "2627"
HORIZON_DAYS = 7


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    now = pd.Timestamp.now()
    today = now.normalize()

    pred = pd.read_csv(PRED, parse_dates=["Date"])
    cutoff = pd.read_csv(DATA, parse_dates=["Date"], usecols=["Date"])["Date"].max()
    if (today - cutoff).days > 10:
        print(f"AVISO: los datos llegan hasta {cutoff.date()}. ¿Falló la descarga?")

    # Solo partidos de los próximos días y con fecha posterior a hoy
    dia = pred["Date"].dt.normalize()
    win = pred[(dia > today) & (dia <= today + pd.Timedelta(days=HORIZON_DAYS))]

    if LOG.exists():
        old = pd.read_csv(LOG, parse_dates=["Date", "generated_at", "data_cutoff"])
        ya = set(zip(old["HomeTeam"], old["AwayTeam"]))
    else:
        old, ya = None, set()

    mask = np.array([(h, a) not in ya
                     for h, a in zip(win["HomeTeam"], win["AwayTeam"])], dtype=bool)
    nuevos = win[mask].copy()
    if nuevos.empty:
        print("Nada nuevo que registrar.")
        return

    nuevos.insert(0, "season", SEASON)
    nuevos.insert(1, "generated_at", now.floor("s"))
    nuevos.insert(2, "data_cutoff", cutoff)
    nuevos.insert(3, "model_commit", git_commit())

    out = nuevos if old is None else pd.concat([old, nuevos], ignore_index=True)
    LOG.parent.mkdir(exist_ok=True)
    out.to_csv(LOG, index=False)
    print(f"{len(nuevos)} predicciones nuevas | total en el registro: {len(out)}")
    print(f"Datos hasta {cutoff.date()} | commit {git_commit()}")


if __name__ == "__main__":
    main()
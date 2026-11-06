from pathlib import Path

import pandas as pd

from fixtures import load_pending

FIX = Path("data/raw/fixtures_2627.csv")
OUT = Path("data/processed/pending_2627.csv")


def main():
    if not FIX.exists():
        raise SystemExit(
            f"Falta {FIX}.\n"
            "Descárgalo a mano desde https://fixturedownload.com/sport/football "
            "(La Liga 2026/27 -> Download fixture as CSV) y guárdalo con ese nombre.")

    df = pd.read_csv("data/processed/laliga_matches.csv",
                     parse_dates=["Date"], dtype={"Season": str})
    played = df[df["Season"] == "2627"]

    fx = load_pending(FIX, played)
    pend = fx[fx["pendiente"]]
    pend.to_csv(OUT, index=False)

    print(f"Calendario: {len(fx)} partidos | jugados: {(~fx['pendiente']).sum()} "
          f"| pendientes: {len(pend)}")
    print(f"Guardado en {OUT}")


if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "PSH", "PSD", "PSA",
    "PSCH", "PSCD", "PSCA",
    "AvgH", "AvgD", "AvgA",
    "AvgCH", "AvgCD", "AvgCA",
    "B365CH", "B365CD", "B365CA",
]


def load_season(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    df = df.dropna(subset=["HomeTeam", "AwayTeam"])  # quita filas vacías
    df = df.reindex(columns=COLUMNS)                 # columnas que faltan -> NaN
    df["Season"] = path.stem.split("_")[1]           # "SP1_2526" -> "2526"
    return df


def main() -> None:
    files = sorted(RAW_DIR.glob("SP1_*.csv"))
    df = pd.concat([load_season(f) for f in files], ignore_index=True)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)

    out_path = PROCESSED_DIR / "laliga_matches.csv"
    df.to_csv(out_path, index=False)

    print(f"Partidos totales: {len(df)}")
    print(f"Rango de fechas: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print("\nPartidos por temporada:")
    print(df["Season"].value_counts().sort_index())
    print("\nValores nulos por columna (solo las que tienen):")
    nulls = df.isna().sum()
    print(nulls[nulls > 0])
    print("\nEquipos:")
    print(sorted(set(df["HomeTeam"]) | set(df["AwayTeam"])))
    print(f"\nPartidos del Betis: {((df['HomeTeam'] == 'Betis') | (df['AwayTeam'] == 'Betis')).sum()}")
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
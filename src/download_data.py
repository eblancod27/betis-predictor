from pathlib import Path
import requests

# Carpeta donde guardaremos los datos originales (sin tocarlos)
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Códigos de temporada: 1617 = temporada 2016/17, ..., 2627 = temporada actual
SEASONS = ["1617", "1718", "1819", "1920", "2021",
           "2122", "2223", "2324", "2425", "2526", "2627"]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"


def download_season(season: str) -> None:
    url = BASE_URL.format(season=season)
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # lanza un error si la descarga falla
    out_path = RAW_DIR / f"SP1_{season}.csv"
    out_path.write_bytes(response.content)
    print(f"OK  {season} -> {out_path} ({len(response.content)} bytes)")


if __name__ == "__main__":
    for season in SEASONS:
        try:
            download_season(season)
        except Exception as e:
            print(f"ERROR {season}: {e}")
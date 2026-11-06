from pathlib import Path
import pandas as pd

# Nombre en fixturedownload -> nombre en football-data (el de nuestros datos)
NAME_MAP = {
    "Athletic Club": "Ath Bilbao",
    "Atlético de Madrid": "Ath Madrid",
    "CA Osasuna": "Osasuna",
    "Celta": "Celta",
    "Deportivo Alavés": "Alaves",
    "Elche CF": "Elche",
    "FC Barcelona": "Barcelona",
    "Getafe CF": "Getafe",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "R. Racing Club": "Santander",
    "Rayo Vallecano": "Vallecano",
    "RC Deportivo": "La Coruna",
    "RCD Espanyol de Barcelona": "Espanol",
    "Real Betis": "Betis",
    "Real Madrid": "Real Madrid",
    "Real Sociedad": "Sociedad",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
}


def load_pending(fixtures_path, played):
    """Partidos de 2026/27 aún no jugados.

    played: DataFrame de partidos ya jugados de la 2627 (columnas HomeTeam, AwayTeam).
    """
    fx = pd.read_csv(fixtures_path)
    cols = {c.lower().replace(" ", ""): c for c in fx.columns}
    home, away, date = cols["hometeam"], cols["awayteam"], cols["date"]

    fx = fx.rename(columns={home: "HomeTeam", away: "AwayTeam", date: "Date"})
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True)

    sin_mapa = (set(fx["HomeTeam"]) | set(fx["AwayTeam"])) - set(NAME_MAP)
    if sin_mapa:
        raise ValueError(f"Equipos sin traducir: {sorted(sin_mapa)}")
    fx["HomeTeam"] = fx["HomeTeam"].map(NAME_MAP)
    fx["AwayTeam"] = fx["AwayTeam"].map(NAME_MAP)

    # En una liga, cada pareja (local, visitante) se juega una vez por temporada:
    # un partido está pendiente si esa pareja no aparece entre los jugados.
    jugados = set(zip(played["HomeTeam"], played["AwayTeam"]))
    fx["pendiente"] = [(h, a) not in jugados for h, a in zip(fx["HomeTeam"], fx["AwayTeam"])]
    return fx[["Date", "HomeTeam", "AwayTeam", "pendiente"]].sort_values("Date").reset_index(drop=True)
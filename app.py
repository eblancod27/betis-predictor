from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data" / "processed"

st.set_page_config(page_title="Betis Predictor", layout="wide")


@st.cache_data
def cargar():
    pred = pd.read_csv(DATA / "predicciones_pendientes.csv", parse_dates=["Date"])
    sim = pd.read_csv(DATA / "simulacion_2627.csv", index_col=0)
    dist = pd.read_csv(DATA / "distribucion_posiciones_2627.csv", index_col=0)
    dist.columns = dist.columns.astype(int)
    return pred, sim, dist


def vista_equipo(pred, team):
    """Partidos pendientes desde el punto de vista del equipo elegido."""
    p = pred[(pred["HomeTeam"] == team) | (pred["AwayTeam"] == team)]
    p = p.sort_values("Date").copy()
    local = p["HomeTeam"] == team
    p["Fecha"] = p["Date"].dt.strftime("%d/%m/%Y")
    p["Rival"] = p["AwayTeam"].where(local, p["HomeTeam"])
    p["Condición"] = local.map({True: "Local", False: "Visitante"})
    p["Victoria"] = p["pH"].where(local, p["pA"])
    p["Empate"] = p["pD"]
    p["Derrota"] = p["pA"].where(local, p["pH"])
    return p[["Fecha", "Rival", "Condición", "Victoria", "Empate", "Derrota"]]


pred, sim, dist = cargar()
equipos = sorted(sim.index)

st.title("Betis Predictor")
st.caption("Probabilidades de LaLiga 2026/27 con un ensamble Elo + Dixon-Coles "
           "y 10.000 simulaciones de la temporada.")

team = st.sidebar.selectbox("Equipo", equipos, index=equipos.index("Betis"))
n_partidos = st.sidebar.slider("Próximos partidos a mostrar", 3, 15, 8)

# --- Resumen del equipo ---
s = sim.loc[team]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Puntos actuales", int(s["pts_actuales"]))
c2.metric("Puntos esperados al final", f"{s['pts_esperados']:.1f}")
c3.metric("Top 4", f"{s['P_top4']:.1%}")
c4.metric("Descenso", f"{s['P_descenso']:.1%}")

# --- Próximos partidos ---
st.subheader(f"Próximos partidos de {team}")
v = vista_equipo(pred, team).head(n_partidos)
st.dataframe(
    v.style.format({"Victoria": "{:.0%}", "Empate": "{:.0%}", "Derrota": "{:.0%}"}),
    hide_index=True, use_container_width=True)

# --- Posición final ---
st.subheader(f"Posición final de {team} (10.000 simulaciones)")
st.bar_chart(dist.loc[team])

# --- Tabla de toda la liga ---
st.subheader("Probabilidades de toda la liga")
tabla = sim.rename(columns={
    "pts_actuales": "Pts actuales", "pts_esperados": "Pts esperados",
    "pos_media": "Posición media", "P_campeon": "Campeón",
    "P_top4": "Top 4", "P_top6": "Top 6", "P_descenso": "Descenso"})
st.dataframe(
    tabla.style.format({"Campeón": "{:.1%}", "Top 4": "{:.1%}", "Top 6": "{:.1%}",
                        "Descenso": "{:.1%}", "Pts esperados": "{:.1f}",
                        "Posición media": "{:.1f}"}),
    use_container_width=True)

with st.expander("Metodología y limitaciones"):
    st.markdown(
        "- **Modelo:** promedio 50/50 de un Elo (con regresión logística) y un "
        "modelo Dixon-Coles con pesos temporales.\n"
        "- **Evaluación (test 2023/24-2025/26):** log-loss 0,9696 frente a 0,9537 "
        "del mercado. La diferencia con el mercado es pequeña pero distinguible.\n"
        "- **Simulación:** fuerzas fijas a fecha 2/11/2026 y desempate aleatorio "
        "(en LaLiga cuenta el enfrentamiento directo).\n"
        "- **Plazas europeas:** top 4 y top 6 son una aproximación.\n"
        "- **Backtest:** 5 temporadas, 100 equipos-temporada. Muestra pequeña y "
        "hiperparámetros elegidos en parte sobre esos mismos datos.\n"
        "- Las probabilidades no incluyen lesiones, alineaciones ni fichajes.")
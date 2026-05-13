import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import distance
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------------------------------------------------

st.set_page_config(
    page_title="centro de inteligencia de la Alianza",
    layout="wide"
)

st.title("centro de inteligencia de la Gran Alianza Shinobi")
st.markdown("rastreo de clones con métricas normalizadas y distancia euclidiana.")


#--------------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

DATA_PATH = "players_data.csv"
df = load_data(DATA_PATH)


#--------------------------------------------------------------

id_cols = ["Nombre", "Edad", "Equipo", "Liga", "Valor_Mercado"]

metric_cols = [
    "Goles",
    "Asistencias",
    "Pases_%",
    "Regates",
    "Recuperaciones",
    "Duelos_Aereos",
    "xG",
    "Potencial"
]

pos_cols = ["Coord_X_Media", "Coord_Y_Media"]

for col in metric_cols + pos_cols + ["Edad", "Valor_Mercado"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=metric_cols)

#--------------------------------------------------------------

@st.cache_data
def minmax_scale(df, cols):
    df_scaled = df.copy()
    for col in cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df_scaled[col + "_norm"] = (df[col] - min_val) / (max_val - min_val)
    return

df_norm = minmax_scale(df, metric_cols)
norm_cols = [c + "_norm" for c in metric_cols]

#--------------------------------------------------------------

def euclidean_distance(p, q):
    return distance.euclidean(p, q)

def get_top_similar_players(
    df_norm,
    target_name,
    norm_cols,
    max_players=5,
    min_age=None,
    max_market_value=None,
    cantera_only=False,
    cantera_age_max=23,
    cantera_potential_min=0.7
):

    target_row = df_norm[df_norm["Nombre"] == target_name]
    if target_row.empty:
        return pd.DataFrame()

    p = target_row[norm_cols].values[0]

    candidates = df_norm.copy()
    candidates = candidates[candidates["Nombre"] != target_name]

    if min_age is not None:
        candidates = candidates[candidates["Edad"] >= min_age]

    if max_market_value is not None:
        candidates = candidates[candidates["Valor_Mercado"] <= max_market_value]

    if cantera_only:
        if "Potencial_norm" in candidates.columns:
            candidates = candidates[
                (candidates["Edad"] <= cantera_age_max) &
                (candidates["Potencial_norm"] >= cantera_potential_min)
            ]

    if candidates.empty:
        return pd.DataFrame()

    distances = []
    for _, row in candidates.iterrows():
        q = row[norm_cols].values
        distances.append(euclidean_distance(p, q))

    candidates["Distancia"] = distances
    candidates = candidates.sort_values("Distancia", ascending=True)

    return candidates.head(max_players)

#--------------------------------------------------------------

st.sidebar.header("panel de control")

target_player = st.sidebar.selectbox(
    "archivo objetivo:",
    sorted(df_norm["Nombre"].unique())
)

min_age = st.sidebar.slider(
    "edad mínima:",
    int(df_norm["Edad"].min()),
    int(df_norm["Edad"].max()),
    int(df_norm["Edad"].min())
)

max_market_value = st.sidebar.slider(
    "valor máximo mercado:",
    float(df_norm["Valor_Mercado"].min()),
    float(df_norm["Valor_Mercado"].max()),
    float(df_norm["Valor_Mercado"].max())
)

st.sidebar.subheader("rastreo de la cantera")
cantera_only = st.sidebar.checkbox("Solo promesas", value=False)
cantera_age_max = st.sidebar.slider("Edad máxima cantera:", 16, 25, 23)
cantera_potential_min = st.sidebar.slider("Potencial mínimo normalizado:", 0.0, 1.0, 0.7, 0.05)

top_n = st.sidebar.slider("Número de clones:", 1, 10, 5)

#--------------------------------------------------------------

similar_players = get_top_similar_players(
    df_norm,
    target_player,
    norm_cols,
    top_n,
    min_age,
    max_market_value,
    cantera_only,
    cantera_age_max,
    cantera_potential_min
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("activo objetivo")
    st.dataframe(df_norm[df_norm["Nombre"] == target_player][id_cols + metric_cols])

with col2:
    st.subheader("clones cercanos")
    if similar_players.empty:
        st.warning("no se encontraron candidatos.")
    else:
        st.dataframe(similar_players[id_cols + metric_cols + ["Distancia"]])

#--------------------------------------------------------------

st.subheader("radar de inteligencia")

def plot_radar(target_row, candidates_df, metric_cols):
    categories = metric_cols
    fig = go.Figure()

    target_values = [target_row[m] for m in categories] + [target_row[categories[0]]]

    fig.add_trace(go.Scatterpolar(
        r=target_values,
        theta=categories + [categories[0]],
        fill='toself',
        name=f"Objetivo: {target_row['Nombre']}",
        line=dict(color="gold")
    ))

    for _, row in candidates_df.iterrows():
        values = [row[m] for m in categories] + [row[categories[0]]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='none',
            name=row["Nombre"]
        ))

    fig.update_layout(showlegend=True, height=600)
    return fig

if not similar_players.empty:
    target_row = df_norm[df_norm["Nombre"] == target_player].iloc[0]
    st.plotly_chart(plot_radar(target_row, similar_players, metric_cols), use_container_width=True)

#--------------------------------------------------------------

st.subheader("mapa de influencia táctica")

if all(col in df_norm.columns for col in pos_cols):
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.kdeplot(
        data=df_norm,
        x="Coord_X_Media",
        y="Coord_Y_Media",
        fill=True,
        cmap="magma",
        levels=100,
        ax=ax
    )

    target_pos = df_norm[df_norm["Nombre"] == target_player][pos_cols].iloc[0]
    ax.scatter(target_pos["Coord_X_Media"], target_pos["Coord_Y_Media"], color="cyan", s=80, label="Objetivo")

    if not similar_players.empty:
        ax.scatter(similar_players["Coord_X_Media"], similar_players["Coord_Y_Media"], color="lime", s=50, label="Clones")

    ax.legend()
    st.pyplot(fig)
else:
    st.info("error")

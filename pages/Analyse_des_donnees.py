# ==========================================================
# ANALYSE DES DONNÉES
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Analyse des données",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================

@st.cache_data
def charger_donnees():
    return pd.read_csv("data_eol.csv", sep=";")

df = charger_donnees()
st.title("📊 Analyse Exploratoire des Données")

st.write("""
Cette page présente les principales analyses réalisées
sur le jeu de données avant l'entraînement
du modèle de Machine Learning.
""")

st.markdown("---")
st.header("📋 Aperçu du Dataset")

st.dataframe(df.head(10), use_container_width=True)

st.write("Dimensions :", df.shape)
c1, c2 = st.columns(2)

with c1:

    st.subheader("Variables")

    st.write(df.columns.tolist())

with c2:

    st.subheader("Types")

    st.dataframe(df.dtypes)
    st.markdown("---")

st.header("📈 Statistiques descriptives")

st.dataframe(df.describe(), use_container_width=True)
st.markdown("---")

st.header("🧹 Valeurs manquantes")

missing = df.isnull().sum()

st.dataframe(missing)

if missing.sum()==0:

    st.success("Aucune valeur manquante.")

else:

    st.error("Des valeurs manquantes existent.")
    st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(

[
"📈 Histogrammes",

"🔥 Corrélations",

"📦 Boxplots",

"📉 Scatter"
]

)
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Histogrammes",
        "🔥 Corrélations",
        "📦 Boxplots",
        "📉 Scatter"
    ]
)
# ==========================================================
# ONGLET 1 : HISTOGRAMMES
# ==========================================================

with tab1:

    st.subheader("Distribution des variables")

    # Choix de la variable
    variable = st.selectbox(
        "Choisir une variable",
        df.select_dtypes(include="number").columns
    )

    # Histogramme interactif
    fig = px.histogram(
        df,
        x=variable,
        nbins=30,
        color_discrete_sequence=["royalblue"]
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=variable,
        yaxis_title="Fréquence"
    )

    st.plotly_chart(fig, use_container_width=True)
    # ==========================================================
# ONGLET 2 : CORRÉLATIONS
# ==========================================================

with tab2:

    st.subheader("Matrice de corrélation")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    # ==========================================================
# ONGLET 3 : BOXPLOTS
# ==========================================================

with tab3:

    st.subheader("Détection des valeurs aberrantes")

    variable = st.selectbox(
        "Choisir une variable",
        df.select_dtypes(include="number").columns,
        key="boxplot"
    )

    fig = px.box(
        df,
        y=variable,
        color_discrete_sequence=["green"]
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    # ==========================================================
# ONGLET 4 : SCATTER PLOT
# ==========================================================

with tab4:

    st.subheader("Relation entre deux variables")

    x = st.selectbox(
        "Variable X",
        df.select_dtypes(include="number").columns,
        key="x"
    )

    y = st.selectbox(
        "Variable Y",
        df.select_dtypes(include="number").columns,
        key="y"
    )

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=y,
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

st.header("📈 Résumé statistique")

st.dataframe(
    df.describe(),
    use_container_width=True
)
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Télécharger le dataset",
    csv,
    "data_eol.csv",
    "text/csv"
)
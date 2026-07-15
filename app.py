# ==========================================================
# PROJET : PRÉDICTION DE LA PRODUCTION D'ÉNERGIE ÉOLIENNE
# Auteur : Bassirou Fall
# Master 2 IMN - UCAD
# ==========================================================

# ==========================================================
# IMPORTATION DES BIBLIOTHÈQUES
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# ==========================================================
# CONFIGURATION DE LA PAGE
# Cette partie doit toujours être placée au début
# ==========================================================

st.set_page_config(
    page_title="Prédiction de la Production Éolienne",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CHARGEMENT DU FICHIER CSS
# Permet de personnaliser le style de l'application
# ==========================================================

def load_css():
    css_path = os.path.join("assets", "style.css")

    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

load_css()

# ==========================================================
# CHARGEMENT DU MODÈLE MACHINE LEARNING
# Le cache évite de recharger le modèle à chaque interaction
# ==========================================================

@st.cache_resource
def charger_modele():

    modele = joblib.load("modele_prediction.pkl")

    return modele

modele = charger_modele()

# ==========================================================
# CHARGEMENT DU DATASET
# Il servira à afficher des informations générales
# ==========================================================

@st.cache_data
def charger_donnees():

    df = pd.read_csv("data_eol.csv", sep=";")

    return df

df = charger_donnees()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌬️ Projet Machine Learning")

# Logo UCAD (facultatif)
logo_ucad = os.path.join("assets", "logo_ucad.png")

if os.path.exists(logo_ucad):
    st.sidebar.image(logo_ucad, width=140)

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Informations")

st.sidebar.write("**Auteur :**")
st.sidebar.write("Bassirou Fall")

st.sidebar.write("**Formation :**")
st.sidebar.write("Master 2 IMN")

st.sidebar.write("**Université :**")
st.sidebar.write("UCAD")

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Modèle")

st.sidebar.success("Régression Linéaire")

st.sidebar.metric(
    label="R²",
    value="96.47 %"
)

st.sidebar.metric(
    label="MAE",
    value="3.98"
)

st.sidebar.metric(
    label="RMSE",
    value="5.00"
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Cette application prédit la production
d'énergie éolienne à partir des
conditions météorologiques.
"""
)

# ==========================================================
# INITIALISATION DE L'HISTORIQUE
# ==========================================================

if "historique" not in st.session_state:

    st.session_state.historique = []

# ==========================================================
# BOUTON DE RÉINITIALISATION
# ==========================================================

if st.sidebar.button("🗑️ Réinitialiser l'historique"):

    st.session_state.historique = []

    st.sidebar.success("Historique supprimé.")
    # ==========================================================
# EN-TÊTE DE L'APPLICATION
# ==========================================================

col1, col2, col3 = st.columns([1,4,1])

with col1:

    logo_ucad = os.path.join("assets", "logo_ucad.jpg")

    if os.path.exists(logo_ucad):
        st.image(logo_ucad, width=120)

with col2:

    st.markdown(
        """
        <h1 style='text-align:center;color:#003366;'>
        🌬️ Système Intelligent de Prédiction de la Production
        d'Énergie Éolienne
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h4 style='text-align:center;color:gray;'>
        Machine Learning • Régression Linéaire • Streamlit
        </h4>
        """,
        unsafe_allow_html=True
    )

with col3:

    logo_senelec = os.path.join("assets", "logo_senelec.png")

    if os.path.exists(logo_senelec):
        st.image(logo_senelec, width=120)

st.markdown("---")
# ==========================================================
# BANNIÈRE
# ==========================================================

image = os.path.join("assets", "eolienne.jpg")

if os.path.exists(image):

    st.image(
        image,
        use_container_width=True
    )
    # ==========================================================
# PRÉSENTATION
# ==========================================================

st.header("📖 Présentation")

st.write(
"""
Cette application a été développée dans le cadre d'un projet de
Machine Learning afin de prédire la production d'énergie éolienne.

Le modèle reçoit plusieurs paramètres météorologiques en entrée,
puis estime automatiquement la production attendue.

L'objectif est d'aider à l'analyse et à la prise de décision
dans le domaine des énergies renouvelables.
"""
)
# ==========================================================
# OBJECTIFS
# ==========================================================

st.markdown("---")

st.header("🎯 Objectifs du projet")

obj1, obj2, obj3 = st.columns(3)

with obj1:

    st.success("""
### 📊 Analyse

Explorer les données
météorologiques.
""")

with obj2:

    st.success("""
### 🤖 Machine Learning

Construire un modèle
de prédiction.
""")

with obj3:

    st.success("""
### 🌐 Déploiement

Créer une application
web avec Streamlit.
""")
    # ==========================================================
# CHIFFRES CLÉS
# ==========================================================

st.markdown("---")

st.header("📈 Chiffres clés")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Nombre d'observations", len(df))

c2.metric("Variables", df.shape[1])

c3.metric("Modèle", "Régression")

c4.metric("R²", "96.47 %")

# ==========================================================
# APERÇU DU DATASET
# ==========================================================

st.markdown("---")

st.header("📋 Aperçu des données")

st.dataframe(df.head(10), use_container_width=True)
# ==========================================================
# TABLEAU DE BORD
# ==========================================================

st.markdown("---")

st.header("📊 Tableau de bord")

# Deux colonnes
col1, col2 = st.columns(2)
with col1:

    st.subheader("Distribution de la production")

    fig = px.histogram(
        df,
        x="Production",
        nbins=30,
        color_discrete_sequence=["#1f77b4"]
    )

    fig.update_layout(
        xaxis_title="Production",
        yaxis_title="Nombre d'observations",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    with col2:

     st.subheader("Distribution de la vitesse du vent")

    fig = px.histogram(
        df,
        x="VitesseVent",
        nbins=30,
        color_discrete_sequence=["green"]
    )

    fig.update_layout(
        xaxis_title="Vitesse du vent",
        yaxis_title="Nombre d'observations",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

st.header("📌 Statistiques rapides")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Production moyenne",
    round(df["Production"].mean(), 2)
)

c2.metric(
    "Production maximale",
    round(df["Production"].max(), 2)
)

c3.metric(
    "Production minimale",
    round(df["Production"].min(), 2)
)

c4.metric(
    "Écart-type",
    round(df["Production"].std(), 2)
)
st.markdown("---")

st.header("🔥 Corrélation des variables")

corr = df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
st.markdown("---")

st.header("📋 Aperçu des données")

st.dataframe(
    df.head(15),
    use_container_width=True
)
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Télécharger le jeu de données",
    csv,
    "data_eol.csv",
    "text/csv"
)
st.markdown("---")

st.caption(
"""
Développé par **Bassirou Fall**

Master 2 Ingénierie Mathématique et Numérique

Université Cheikh Anta Diop de Dakar

Projet Machine Learning • Streamlit • Scikit-Learn
"""
)

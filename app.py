# ==========================================================
# APPLICATION STREAMLIT
# Prédiction de la production d'énergie éolienne
# Auteur : Bassirou Fall
# Master 2 IMN - UCAD
# ==========================================================

# ==========================
# IMPORTATION DES BIBLIOTHÈQUES
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ==========================
# CONFIGURATION DE LA PAGE
# ==========================

st.set_page_config(
    page_title="Prédiction Éolienne",
    page_icon="🌬️",
    layout="wide"
)

# ==========================
# STYLE PERSONNALISÉ
# ==========================

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 10px 15px;
    }
    div[data-testid="stMetricValue"] {
        color: #1f4e8c;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# CHARGEMENT DU MODÈLE (avec cache + gestion d'erreur)
# ==========================

@st.cache_resource
def charger_modele():
    try:
        return joblib.load("modele_prediction.pkl")
    except FileNotFoundError:
        return None

modele = charger_modele()

# ==========================
# TITRE
# ==========================

st.title("🌬️ Prédiction de la Production d'Énergie Éolienne")

st.markdown("""
Cette application utilise un modèle de **Régression Linéaire**
pour prédire la production d'énergie à partir des conditions
météorologiques.
""")

if modele is None:
    st.error(
        "⚠️ Le fichier `modele_prediction.pkl` est introuvable. "
        "Vérifiez qu'il se trouve bien dans le même dossier que cette application."
    )
    st.stop()

st.divider()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("À propos")

st.sidebar.success("Projet Machine Learning")

st.sidebar.write("Université Cheikh Anta Diop")

st.sidebar.write("Master 2 IMN")

st.sidebar.write("Auteur : Bassirou Fall")

st.sidebar.divider()

st.sidebar.subheader("Performance du modèle")

st.sidebar.metric("Modèle", "Régression Linéaire")

st.sidebar.metric("R²", "0.9647")

st.sidebar.metric("MAE", "3.98")

st.sidebar.metric("RMSE", "5.00")

st.sidebar.divider()

if st.sidebar.button("🗑️ Réinitialiser l'historique"):
    st.session_state.historique = []
    st.sidebar.success("Historique réinitialisé !")

# ==========================
# INITIALISATION DE L'HISTORIQUE
# ==========================

if "historique" not in st.session_state:
    st.session_state.historique = []

# ==========================
# ONGLETS PRINCIPAUX
# ==========================

onglet_prediction, onglet_historique = st.tabs(["🔮 Prédiction", "📊 Historique & Tendances"])

# ===================================================
# ONGLET 1 : PRÉDICTION
# ===================================================

with onglet_prediction:

    col1, col2 = st.columns([2, 1])

    with col1:

        st.subheader("Saisir les paramètres")

        vitesse = st.slider(
            "Vitesse du vent (m/s)",
            min_value=5.0,
            max_value=20.0,
            value=12.5,
            help="Vitesse moyenne du vent mesurée à hauteur de moyeu."
        )

        direction = st.slider(
            "Direction du vent (°)",
            min_value=0,
            max_value=360,
            value=180,
            help="Direction du vent en degrés (0° = Nord, 90° = Est, etc.)"
        )

        temperature = st.slider(
            "Température (°C)",
            min_value=15.0,
            max_value=35.0,
            value=25.0,
            help="Température de l'air ambiant."
        )

        pression = st.slider(
            "Pression (hPa)",
            min_value=1000.0,
            max_value=1025.0,
            value=1013.0,
            help="Pression atmosphérique au niveau du site."
        )

        humidite = st.slider(
            "Humidité (%)",
            min_value=40.0,
            max_value=90.0,
            value=65.0,
            help="Taux d'humidité relative de l'air."
        )

    with col2:

        st.subheader("Informations")

        st.metric("Modèle", "Régression")
        st.metric("Variables", "5")
        st.metric("R²", "96.47 %")
        st.metric("MAE", "3.98")
        st.metric("RMSE", "5.00")

    st.divider()

    if st.button("🔍 Prédire", type="primary"):

        X = np.array([[vitesse, direction, temperature, pression, humidite]])

        prediction = modele.predict(X)[0]

        # Comparaison avec la moyenne de l'historique existant (avant ajout)
        if st.session_state.historique:
            moyenne_precedente = pd.DataFrame(st.session_state.historique)["Production (kW)"].mean()
        else:
            moyenne_precedente = None

        st.success(f"⚡ Production prédite : {prediction:.2f} kW")

        col_gauge, col_resume = st.columns([1.3, 1])

        with col_gauge:

            st.subheader("Jauge de la production")

            mode_jauge = "gauge+number+delta" if moyenne_precedente is not None else "gauge+number"

            fig = go.Figure(go.Indicator(
                mode=mode_jauge,
                value=prediction,
                number={"suffix": " kW"},
                delta={"reference": moyenne_precedente, "valueformat": ".2f"} if moyenne_precedente is not None else None,
                title={"text": "Production (kW)"},
                gauge={
                    "axis": {"range": [0, 130]},
                    "bar": {"color": "black"},
                    "steps": [
                        {"range": [0, 40], "color": "#f8d7da"},
                        {"range": [40, 80], "color": "#fff3cd"},
                        {"range": [80, 130], "color": "#d4edda"}
                    ]
                }
            ))
            fig.update_layout(height=320, margin=dict(t=50, b=10))

            st.plotly_chart(fig, use_container_width=True)

        with col_resume:

            st.subheader("Résumé des données")

            resultat = pd.DataFrame({
                "Variable": ["Vitesse du vent", "Direction", "Température", "Pression", "Humidité"],
                "Valeur": [vitesse, direction, temperature, pression, humidite]
            })

            st.dataframe(resultat, use_container_width=True, hide_index=True)

        # Ajout à l'historique
        st.session_state.historique.append({
            "Vitesse": vitesse,
            "Direction": direction,
            "Température": temperature,
            "Pression": pression,
            "Humidité": humidite,
            "Production (kW)": round(prediction, 2)
        })

        st.info("➡️ Consultez l'onglet **Historique & Tendances** pour voir l'évolution de vos prédictions.")

# ===================================================
# ONGLET 2 : HISTORIQUE ET TENDANCES
# ===================================================

with onglet_historique:

    if not st.session_state.historique:

        st.info("Aucune prédiction enregistrée pour le moment. Rendez-vous dans l'onglet **Prédiction** pour commencer.")

    else:

        historique = pd.DataFrame(st.session_state.historique)

        st.subheader("Historique des prédictions")
        st.dataframe(historique, use_container_width=True, hide_index=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Nombre de prédictions", len(historique))
        col_b.metric("Production moyenne", f"{historique['Production (kW)'].mean():.2f} kW")
        col_c.metric("Production max", f"{historique['Production (kW)'].max():.2f} kW")

        st.subheader("Évolution de la production prédite")

        fig_tendance = go.Figure()
        fig_tendance.add_trace(go.Scatter(
            y=historique["Production (kW)"],
            mode="lines+markers",
            line=dict(color="#1f4e8c", width=2),
            marker=dict(size=8),
            name="Production"
        ))
        fig_tendance.update_layout(
            xaxis_title="N° de prédiction",
            yaxis_title="Production (kW)",
            height=350,
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(fig_tendance, use_container_width=True)

        csv = historique.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Télécharger l'historique (CSV)",
            csv,
            "historique_predictions.csv",
            "text/csv"
        )

st.divider()

st.caption("Application développée avec Streamlit | Machine Learning | Régression Linéaire")
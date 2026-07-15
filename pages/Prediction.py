# ==========================================================
# PRÉDICTION DE LA PRODUCTION D'ÉNERGIE ÉOLIENNE
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Prédiction",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# CHARGEMENT DU MODÈLE
# ==========================================================

@st.cache_resource
def charger_modele():
    return joblib.load("modele_prediction.pkl")

modele = charger_modele()

# ==========================================================
# TITRE
# ==========================================================

st.title("🤖 Prédiction de la Production Éolienne")

st.write("""
Renseignez les paramètres météorologiques afin de prédire
la production d'énergie éolienne.
""")

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Modèle", "Régression")
c2.metric("Variables", "5")
c3.metric("R²", "96.47 %")
c4.metric("RMSE", "5.00")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌤️ Paramètres météorologiques")

    vitesse = st.slider(
        "🌬️ Vitesse du vent (m/s)",
        5.0,
        20.0,
        12.5
    )

    direction = st.slider(
        "🧭 Direction du vent (°)",
        0,
        360,
        180
    )

    temperature = st.slider(
        "🌡️ Température (°C)",
        15.0,
        35.0,
        25.0
    )

    pression = st.slider(
        "🌍 Pression (hPa)",
        1000.0,
        1025.0,
        1013.0
    )

    humidite = st.slider(
        "💧 Humidité (%)",
        40.0,
        90.0,
        65.0
    )

with col2:
    st.subheader("📋 Résumé")

    resume = pd.DataFrame({
        "Variable": [
            "Vitesse",
            "Direction",
            "Température",
            "Pression",
            "Humidité"
        ],
        "Valeur": [
            vitesse,
            direction,
            temperature,
            pression,
            humidite
        ]
    })

    st.dataframe(
        resume,
        use_container_width=True
    )

st.markdown("---")

# ==========================================================
# PRÉDICTION (déclenchée par le bouton)
# ==========================================================

if st.button("🚀 Lancer la prédiction"):

    X = np.array([[
        vitesse,
        direction,
        temperature,
        pression,
        humidite
    ]])

    prediction = modele.predict(X)[0]

    # ==========================================================
    # JAUGE DE LA PRODUCTION
    # ==========================================================

    st.subheader("📊 Niveau de production")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        number={"suffix": " kW"},
        title={"text": "Production estimée"},
        gauge={
            "axis": {"range": [0, 130]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 40], "color": "#ffcccc"},
                {"range": [40, 80], "color": "#fff2cc"},
                {"range": [80, 130], "color": "#d9ead3"}
            ]
        }
    ))

    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )

    st.success(
        f"⚡ Production prédite : {prediction:.2f} kW"
    )

    st.subheader("📌 Interprétation")

    if prediction < 40:
        st.error("🔴 Production faible")
    elif prediction < 80:
        st.warning("🟠 Production moyenne")
    else:
        st.success("🟢 Production élevée")

    # ==========================================================
    # GRAPHIQUE DES VARIABLES
    # ==========================================================

    st.subheader("📈 Paramètres utilisés")

    variables = pd.DataFrame({
        "Variable": [
            "Vitesse",
            "Direction",
            "Température",
            "Pression",
            "Humidité"
        ],
        "Valeur": [
            vitesse,
            direction,
            temperature,
            pression,
            humidite
        ]
    })

    fig_bar = px.bar(
        variables,
        x="Variable",
        y="Valeur",
        text="Valeur",
        color="Valeur",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    # ==========================================================
    # RADAR
    # ==========================================================

    st.subheader("🕸️ Profil météorologique")

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=[
            vitesse,
            direction / 10,
            temperature,
            pression / 50,
            humidite
        ],
        theta=[
            "Vitesse",
            "Direction",
            "Température",
            "Pression",
            "Humidité"
        ],
        fill='toself',
        name="Conditions"
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        ),
        showlegend=False
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # ==========================================================
    # HISTORIQUE
    # ==========================================================

    if "historique" not in st.session_state:
        st.session_state.historique = []

    st.session_state.historique.append({
        "Vitesse": vitesse,
        "Direction": direction,
        "Température": temperature,
        "Pression": pression,
        "Humidité": humidite,
        "Production": round(prediction, 2)
    })

# ==========================================================
# AFFICHAGE DE L'HISTORIQUE (persiste entre les clics)
# ==========================================================

if "historique" in st.session_state and len(st.session_state.historique) > 0:

    st.markdown("---")
    st.subheader("📜 Historique des prédictions")

    historique = pd.DataFrame(st.session_state.historique)

    st.dataframe(
        historique,
        use_container_width=True
    )

    csv = historique.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Télécharger l'historique",
        csv,
        "historique.csv",
        "text/csv"
    )

    if st.button("🗑️ Vider l'historique"):
        st.session_state.historique = []
        st.rerun()

st.markdown("---")

st.caption("""
Développé par Bassirou Fall

Master 2 IMN

Université Cheikh Anta Diop

Projet Machine Learning

Régression Linéaire
""")
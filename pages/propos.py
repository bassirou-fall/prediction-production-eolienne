# ==========================================================
# PAGE À PROPOS DU PROJET
# ==========================================================

import streamlit as st


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="À propos",

    page_icon="ℹ️",

    layout="wide"

)
# ==========================================================
# TITRE
# ==========================================================

st.title(
    "ℹ️ À propos du projet"
)

st.markdown("---")
st.header("🌬️ Présentation")

st.write(
"""
Ce projet porte sur la prédiction de la production
d'énergie éolienne à l'aide des techniques de Machine Learning.

L'objectif principal est de construire un modèle capable
d'estimer la quantité d'énergie produite à partir
des paramètres météorologiques.

Les données utilisées concernent principalement :

- La vitesse du vent
- La direction du vent
- La température
- La pression atmosphérique
- L'humidité

Le modèle retenu après comparaison est la
Régression Linéaire.
"""
)
st.markdown("---")

st.header("👨‍💻 Auteur")

col1, col2 = st.columns(2)


with col1:

    st.info(
"""
**Nom :** Bassirou Fall

**Formation :**
Master 2 Ingénierie Mathématique et Numérique (IMN)

**Université :**
Université Cheikh Anta Diop de Dakar

"""
    )


with col2:

    st.success(
"""
**Domaine :**

Data Science

Machine Learning

Intelligence Artificielle

Développement Web

"""
    )
    st.markdown("---")

st.header("🎯 Objectifs")

objectif1, objectif2, objectif3 = st.columns(3)


with objectif1:

    st.write(
"""
### 📊 Analyse

Explorer et comprendre
les données météorologiques.
"""
    )


with objectif2:

    st.write(
"""
### 🤖 Modélisation

Construire et comparer
plusieurs modèles ML.
"""
    )


with objectif3:

    st.write(
"""
### 🌐 Déploiement

Créer une application
web avec Streamlit.
"""
    )
    st.markdown("---")

st.header("🔄 Pipeline du projet")


st.code(
"""
Collecte des données

        ↓

Analyse exploratoire (EDA)

        ↓

Nettoyage des données

        ↓

Prétraitement

(StandardScaler / OneHotEncoder)

        ↓

Séparation Train / Test

        ↓

Entraînement des modèles

        ↓

Évaluation

(MAE, RMSE, R²)

        ↓

Choix du meilleur modèle

        ↓

Sauvegarde (.pkl)

        ↓

Application Streamlit

        ↓

Prédiction en temps réel
"""
)
st.markdown("---")

st.header("🤖 Modèles évalués")


modeles = {

    "Régression Linéaire":0.9647,

    "Decision Tree":0.9260,

    "Random Forest":0.9626,

    "XGBoost":0.9644

}


st.dataframe(

    modeles,

    use_container_width=True

)
st.markdown("---")

st.header("🏆 Modèle final")


c1,c2,c3 = st.columns(3)


c1.metric(

    "Modèle",

    "Régression Linéaire"

)


c2.metric(

    "R²",

    "96.47 %"

)


c3.metric(

    "RMSE",

    "5.00"

)
st.markdown("---")

st.header("🛠️ Technologies")


col1,col2,col3 = st.columns(3)


with col1:

    st.write(
"""
🐍 Python

- Pandas

- NumPy

- Scikit-Learn
"""
    )


with col2:

    st.write(
"""
📊 Data Visualisation

- Matplotlib

- Seaborn

- Plotly
"""
    )


with col3:

    st.write(
"""
🌐 Déploiement

- Streamlit

- Joblib

- Git
"""
    )
    st.markdown("---")

st.header("✅ Conclusion")


st.success(
"""
Ce projet montre l'utilisation complète d'un
cycle Machine Learning :

de l'analyse des données jusqu'au déploiement
d'une application interactive permettant
la prédiction en temps réel.
"""
)
st.markdown("---")

st.caption(
"""
Projet réalisé par Bassirou Fall

Master 2 IMN - UCAD

Machine Learning | Data Science | Streamlit
"""
)
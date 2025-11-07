# app.py
import streamlit as st
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *
from optimisation_edt_v2 import build_model, solve_model, extract_solution  # Adapter selon ton projet

st.title("Optimisation Staffing Projects - Saisie interactive")

# -----------------------
# Section 1 : Paramètres généraux
# -----------------------
st.header("1. Paramètres généraux")
H = st.number_input("Horizon (nombre de jours)", min_value=1, value=10, step=1)

# -----------------------
# Section 2 : Qualifications
# -----------------------
st.header("2. Qualifications")
qualifications_input = st.text_input("Entrez les qualifications séparées par des virgules", "Dev,QA,PM")
J = [q.strip() for q in qualifications_input.split(",")]

# -----------------------
# Section 3 : Personnel
# -----------------------
st.header("3. Personnel")
n_staff = st.number_input("Nombre de membres du personnel", min_value=1, value=2, step=1)

I = []
eta = defaultdict(lambda: defaultdict(int))
v = defaultdict(lambda: defaultdict(int))

for i in range(n_staff):
    st.subheader(f"Membre {i+1}")
    name = st.text_input(f"Nom du membre {i+1}", value=f"Personne_{i+1}", key=f"name_{i}")
    I.append(name)
    quals = st.multiselect(f"Qualifications de {name}", options=J, key=f"qual_{i}")
    for j in J:
        eta[name][j] = 1 if j in quals else 0
    vacs = st.text_input(f"Jours de vacances de {name} séparés par des virgules (laisser vide si aucun)", key=f"vac_{i}")
    vacations_set = set(int(d.strip()) for d in vacs.split(",") if d.strip())
    for t in range(1, H+1):
        v[name][t] = 0 if t in vacations_set else 1

# -----------------------
# Section 4 : Projets
# -----------------------
st.header("4. Projets")
n_jobs = st.number_input("Nombre de projets", min_value=1, value=2, step=1)

K = []
mu = defaultdict(lambda: defaultdict(int))
gain = {}
due = {}
pen = {}

for k_i in range(n_jobs):
    st.subheader(f"Projet {k_i+1}")
    k = st.text_input(f"Nom du projet {k_i+1}", value=f"Projet_{k_i+1}", key=f"proj_{k_i}")
    K.append(k)
    gain[k] = st.number_input(f"Gain du projet {k}", value=100, step=1, key=f"gain_{k_i}")
    due[k]  = st.number_input(f"Date limite (jour) du projet {k}", min_value=1, max_value=H, value=H, step=1, key=f"due_{k_i}")
    pen[k]  = st.number_input(f"Pénalité quotidienne pour le projet {k}", value=10, step=1, key=f"pen_{k_i}")
    
    st.text(f"Jours requis par qualification pour {k} :")
    for j in J:
        mu[k][j] = st.number_input(f"{j} pour {k}", min_value=0, value=1, step=1, key=f"mu_{k}_{j}")

# -----------------------
# Section 5 : Objectif et Solveur
# -----------------------
st.header("5. Objectif et Solveur")
OBJECTIVE = st.selectbox("Objectif", ["profit", "nb_projects", "on_time", "combo"])
if OBJECTIVE == "combo":
    W_PROFIT = st.number_input("Poids profit", value=1.0, step=0.1)
    W_NPROJ  = st.number_input("Poids nb_projects", value=0.0, step=0.1)
    W_ONTIME = st.number_input("Poids on_time", value=0.0, step=0.1)
else:
    W_PROFIT, W_NPROJ, W_ONTIME = 1.0, 0.0, 0.0

SOLVER_NAME = st.selectbox("Solveur", ["glpk", "cbc"])
TEE = st.checkbox("Afficher les logs du solveur", value=False)
TIMELIMIT = st.number_input("Temps limite (s, optionnel)", min_value=0, value=0, step=10)
TIMELIMIT = TIMELIMIT if TIMELIMIT > 0 else None

# -----------------------
# Section 6 : Résolution
# -----------------------
if st.button("Résoudre le modèle"):
    with st.spinner("Résolution en cours..."):
        try:
            model = build_model(H, I, J, K, eta, v, mu, gain, due, pen,
                                objective=OBJECTIVE,
                                w_profit=W_PROFIT, w_nproj=W_NPROJ, w_ontime=W_ONTIME)
            solve_model(model, solver_name=SOLVER_NAME, tee=TEE, timelimit=TIMELIMIT)
            sol = extract_solution(model)
            st.success(f"Résolution terminée. Objectif = {sol['objective']:.2f}")

            # ----- Affichage des résultats -----
            st.header("Résultats")
            df_proj = pd.DataFrame({
                "Projet": list(sol["projects_done"].keys()),
                "Réalisé": list(sol["projects_done"].values()),
                "Jour complétion": [sol["completion_day"][k] for k in sol["projects_done"]],
                "À l'heure": [sol["on_time"][k] for k in sol["projects_done"]]
            })
            st.dataframe(df_proj)

            # Graphiques
            st.subheader("Histogramme des jours de complétion")
            plt.figure(figsize=(8,4))
            df_proj[df_proj["Réalisé"]==1]["Jour complétion"].hist(bins=range(1,H+2))
            plt.xlabel("Jour")
            plt.ylabel("Nombre de projets complétés")
            st.pyplot(plt)

            st.subheader("Projets réalisés vs non réalisés")
            plt.figure(figsize=(6,4))
            df_proj["Réalisé"].value_counts().plot(kind="bar", color=["red","green"])
            plt.xticks([0,1], ["Non réalisé","Réalisé"], rotation=0)
            plt.ylabel("Nombre de projets")
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Erreur lors de la résolution : {e}")

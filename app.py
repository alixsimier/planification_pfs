import streamlit as st
import tempfile
import matplotlib.pyplot as plt
from pyomo.environ import value
from optimisation_edt import solve_model, calculs
import numpy as np
import pandas as pd

from visualisation import plot_planning_complet, projet_color_dict

st.title("CompuOpti")

### CONFIGURATION DE L'INSTANCE

instance_file = st.file_uploader("Choisir un fichier d'instance JSON", type="json")

st.subheader("Objectif principal")
# Objectif principal
objective_main = st.selectbox("", ["profit", "personne", "duree", "projets"])

# Objectifs secondaires
st.subheader("Objectifs secondaires")
obj_secondaires = {}
if st.checkbox("Profit", key="sec_profit"):
    obj_secondaires["profit"] = st.number_input("Profit minimal", min_value=0, max_value=1000, value=1, step=1, key="epsilon_profit")

if st.checkbox("Nombre de projets par personne", key="sec_personne"):
    obj_secondaires["personne"] = st.number_input("Nombre moyen maximal de projets par personne", min_value=0, max_value=50, value=1, step=1, key="epsilon_personne")

if st.checkbox("Durée", key="sec_duree"):
    obj_secondaires["duree"] = st.number_input("Durée moyenne maximale d'un projet", min_value=0, max_value=50, value=1, step=1, key="epsilon_duree")

if st.checkbox("Projets", key="sec_projets"):
    obj_secondaires["projets"] = st.number_input("Nombre minimal de projets", min_value=0, max_value=50, value=1, step=1, key="epsilon_projets")

### RÉSOLUTION
if instance_file is not None and st.button("Lancer la résolution"):
    st.info("Résolution du modèle en cours...")

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(instance_file.read())
        tmp_path = tmp.name


    CONFIG = {
        "instance_path": tmp_path,
        "obj_principale": objective_main,
        "obj_secondaires": obj_secondaires,
        "solver": "gurobi",
        "tee": False
    }

    # Résolution du modèle
    m, results = solve_model(CONFIG)

    if str(results.solver.termination_condition) == "optimal":
        st.success("Résolution terminée !")

        # --- Affichage des valeurs des objectifs ---
        st.subheader("Valeurs des objectifs")
        prof, personne, duree, projet, retard, projets_faits_details = calculs(m, results)
        stats_dict = {
            "Indicateur": [
                "Profit total réalisé",
                "Nombre moyen de projet par personne",
                "Durée moyenne d'un projet",
                "Nombre de projets réalisés",
                "Nombre de projets rendus en retard"
            ],
            "Valeur": [
                str(prof),
                str(round(personne, 2)),
                str(round(duree, 2)),
                str(projet),
                str(retard)
            ]
        }

        # Convertir en DataFrame
        stats_df = pd.DataFrame(stats_dict)

        # Afficher dans Streamlit
        st.table(stats_df)

        # --- Préparation du planning pour le graphique ---
        S, P, T = list(m.S), list(m.P), list(m.T)
        projets_list = sorted({p for _, _, p, _ in m.SQPT})
        projet_to_idx = {p: i for i, p in enumerate(projets_list)}
        color_dict = projet_color_dict(P)
        planning_data = []

        for s in S:
            sdic = {"personne": s, "taches": []} 
            taches = []
            proj = None
            act = None
            for t in m.T:
                projet_trouve = None
                for q in m.Q:
                    for p in m.P:
                        if (s, q, p, t) in m.SQPT and value(m.lmbda[s, q, p, t]) > 0.5:
                            projet_trouve = p
                            break
                    if projet_trouve:
                        break

                if projet_trouve is not None:
                    if projet_trouve == proj:
                        act["t_end"] = t
                    else:
                        if act is not None:
                            taches.append(act)
                        proj = projet_trouve
                        act = {"projet": proj, "t_start": t, "t_end": t}
                else:
                    if act is not None:
                        taches.append(act)
                        act = None
                        proj = None
            if act is not None:
                taches.append(act)

            sdic["taches"] = taches
            planning_data.append(sdic)
        
        print(planning_data)

        plot_planning_complet(
            projets_faits_details=projets_faits_details, 
            S=S, 
            project_colors_dict=color_dict, 
            planning_data=planning_data)
    
    else:
        st.write(f"Aucune solution n'existe. Le problème est {results.solver.termination_condition}.")

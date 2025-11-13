import streamlit as st
import tempfile
import matplotlib.pyplot as plt
from pyomo.environ import value
from optimisation_edt import solve_model, calculs
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

st.title("Staffing Projects Interactive Dashboard")

### CONFIGURATION DE L'INSTANCE

instance_file = st.file_uploader("Choisir un fichier d'instance JSON", type="json")

# Objectif principal
objective_main = st.selectbox("Objectif principal", ["profit", "personne", "duree", "projets"])

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
        prof, personne, duree, projet = calculs(m, results)
        st.write(f"Profit total réalisé: {prof}")
        st.write(f"Nombre moyen de projet par personne : {personne}")
        st.write(f"Durée moyenne d'un projet : {duree}")
        st.write(f"Nombre de projets réalisés : {projet}")

        # --- Préparation du planning pour le graphique ---
        S, P, T = list(m.S), list(m.P), list(m.T)
        projets_list = list(P)
        projet_to_idx = {p: i for i, p in enumerate(projets_list)}

        planning_data_num = -1 * np.ones((len(S), len(T)))  # -1 = vacances / aucun projet
        for i, s in enumerate(S):
            for j, t in enumerate(T):
                for q in m.Q:
                    for p in P:
                        if (s, q, p, t) in m.SQPT and value(m.lmbda[s, q, p, t]) > 0.5:
                            planning_data_num[i, j] = projet_to_idx[p]

        cmap = plt.get_cmap("tab20", len(projets_list))
        new_colors = np.vstack(([1, 1, 1, 1], cmap(np.arange(len(projets_list)))))
        new_cmap = mcolors.ListedColormap(new_colors)
        bounds = np.arange(-1, len(projets_list) + 1)
        norm = mcolors.BoundaryNorm(bounds, new_cmap.N)

        fig, ax = plt.subplots(figsize=(10, 5))
        cax = ax.imshow(planning_data_num, aspect='auto', cmap=new_cmap, norm=norm, origin='lower')

        ax.set_xlabel("Jour")
        ax.set_ylabel("Personne")
        ax.set_yticks(range(len(S)))
        ax.set_yticklabels(S)
        ax.set_title("Planning des projets par personne")

        # Légende manuelle
        patches = [mpatches.Patch(color=new_cmap(i + 1), label=p) for i, p in enumerate(projets_list)]
        patches.insert(0, mpatches.Patch(color="white", label="Vacances"))
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Projets")

        st.pyplot(fig)
    
    else:
        st.write(f"Aucune solution n'existe. Le problème est {results.solver.termination_condition}.")

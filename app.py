import streamlit as st
import tempfile
import matplotlib.pyplot as plt
from pyomo.environ import value
from optimisation_edt import solve_model
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

st.title("Staffing Projects Interactive Dashboard")

# --- CONFIGURATION DE L'INSTANCE ---
instance_file = st.file_uploader("Choisir un fichier d'instance JSON", type="json")

# Objectif principal
objective_main = st.selectbox("Objectif principal", ["profit", "compacite", "duree"])

# Objectifs secondaires
st.subheader("Objectifs secondaires")
obj_secondaires = {}
if st.checkbox("Profit", key="sec_profit"):
    obj_secondaires["profit"] = st.number_input("Epsilon profit", min_value=0, max_value=1000, value=1, step=1, key="epsilon_profit")

if st.checkbox("Compacité", key="sec_compacite"):
    obj_secondaires["compacite"] = st.number_input("Epsilon compacité", min_value=0, max_value=50, value=1, step=1, key="epsilon_compacite")

if st.checkbox("Durée", key="sec_duree"):
    obj_secondaires["duree"] = st.number_input("Epsilon durée", min_value=0, max_value=50, value=1, step=1, key="epsilon_duree")


# --- BOUTON DE RESOLUTION ---
if instance_file is not None and st.button("Lancer la résolution"):
    st.info("Résolution du modèle en cours...")

    # Sauvegarder temporairement le fichier pour que load_instance puisse le lire
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
    st.success("Résolution terminée !")

    # --- Affichage des valeurs des objectifs ---
    st.subheader("Valeurs des objectifs")
    st.write(f"Profit total réalisé: {value(m.expr_profit):.2f}")
    st.write(f"Nombre moyen de projet par personne : {value(m.expr_compacite):.2f}")
    st.write(f"Durée moyenne d'un projet (inclus les projets non réalisés) : {value(m.expr_duree):.2f}")

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

    # --- Colormap avec blanc pour vacances ---
    cmap = plt.get_cmap("tab20", len(projets_list))
    new_colors = np.vstack(([1, 1, 1, 1], cmap(np.arange(len(projets_list)))))
    new_cmap = mcolors.ListedColormap(new_colors)
    bounds = np.arange(-1, len(projets_list) + 1)
    norm = mcolors.BoundaryNorm(bounds, new_cmap.N)

    # --- Affichage du planning ---
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

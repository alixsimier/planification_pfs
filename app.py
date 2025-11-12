import streamlit as st
import tempfile
import matplotlib.pyplot as plt
from pyomo.environ import value
from optimisation_edt import solve_model  # ton fichier avec solve_model
import numpy as np 
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import value

st.title("Staffing Projects Interactive Dashboard")

# --- CONFIGURATION DE L'INSTANCE ---
instance_file = st.file_uploader("Choisir un fichier d'instance JSON", type="json")

objective = st.selectbox("Objectif", ["profit", "projets", "multi"])
w_profit = st.slider("Poids profit (si multi)", 0.0, 1.0, 0.8)
max_projet_par_pers = st.slider("Charge max par personne", 1, 10, 5)

# --- BOUTON DE RESOLUTION ---
if instance_file is not None and st.button("Lancer la résolution"):
    st.info("Résolution du modèle en cours...")
    
    # Sauvegarder temporairement le fichier pour que load_instance puisse le lire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(instance_file.read())
        tmp_path = tmp.name

    CONFIG = {
        "instance_path": tmp_path,
        "objective": objective,
        "w_profit": w_profit,
        "contrainte_charge": max_projet_par_pers,
        "solver": "gurobi",
        "tee": False
    }

    results = solve_model(CONFIG)
    st.success("Résolution terminée !")
    
    # --- EXTRACTION ET GRAPHIQUES ---
    m, results = solve_model(CONFIG)
    
    S, P, T = list(m.S), list(m.P), list(m.T)

    projets_list = list(P)
    projet_to_idx = {p: i for i, p in enumerate(projets_list)}

    planning_data_num = -1 * np.ones((len(S), len(T)))  # -1 = vacances / aucun projet

    for i, s in enumerate(S):
        for j, t in enumerate(T):
            for q in m.Q:
                for p in P:
                    if (s,q,p,t) in m.SQPT and value(m.lmbda[s,q,p,t]) > 0.5:
                        planning_data_num[i,j] = projet_to_idx[p]

    # --- Colormap avec blanc pour vacances ---
    cmap = plt.get_cmap("tab20", len(projets_list))
    # Ajouter blanc pour -1
    new_colors = np.vstack(([1,1,1,1], cmap(np.arange(len(projets_list)))))
    new_cmap = mcolors.ListedColormap(new_colors)

    # Norme : -1 → blanc, 0..N → projets
    bounds = np.arange(-1, len(projets_list)+1)
    norm = mcolors.BoundaryNorm(bounds, new_cmap.N)

    # --- Affichage ---
    fig, ax = plt.subplots(figsize=(10,5))
    cax = ax.imshow(planning_data_num, aspect='auto', cmap=new_cmap, norm=norm, origin='lower')

    ax.set_xlabel("Jour")
    ax.set_ylabel("Personne")
    ax.set_yticks(range(len(S)))
    ax.set_yticklabels(S)
    ax.set_title("Planning des projets par personne")

    # Légende manuelle (ignorer -1)
    patches = [mpatches.Patch(color=new_cmap(i+1), label=p) for i, p in enumerate(projets_list)]
    patches.insert(0, mpatches.Patch(color="white", label="Vacances"))  # ajouter blanc
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Projets")

    st.pyplot(fig)
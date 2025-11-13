import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st

def projet_color_dict(P):
    n_projets = len(P)
    cmap_base = plt.get_cmap("tab20", n_projets)  # palette tab20
    color_dict = {p: (*cmap_base(i)[:3], 0.6) for i, p in enumerate(P)}  # RGBA avec alpha=0.6
    return color_dict

def plot_planning_complet(projets_faits_details, S, project_colors_dict, planning_data):
    """
    Affiche les créneaux par projet et le planning par personne avec couleurs cohérentes.
    projets_faits_details : dict avec t_min, t_max, due_date pour chaque projet
    planning_data : liste de dicts avec les tâches par personne
    S : liste des noms de personnes
    project_colors_dict : dict {nom_projet: couleur RGBA}
    """
    projets_list = list(projets_faits_details.keys())
    n_projets = len(projets_list)
    
    # Création de la figure avec 2 sous-graphiques
    # fig, axes = plt.subplots(2, 1, figsize=(14, 8 + n_projets*0.4), constrained_layout=True)
    # ax1, ax2 = axes
    
    # -------------------
    # Graphique 1 : créneaux par projet
    # -------------------
    hauteur = 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8 + n_projets*0.4), constrained_layout=True, sharex=True)

    # -------------------
    # Graphique 1 : créneaux par projet
    # -------------------
    for i, p in enumerate(projets_list):
        details = projets_faits_details[p]
        t_min = details['t_min']
        t_max = details['t_max'] + 1
        due = details.get('due_date', t_max) + 1
        color = project_colors_dict.get(p, (0.7, 0.7, 0.7, 0.6))  # gris si non défini

        t = np.arange(t_min, t_max + 1)
        f = np.ones_like(t) * hauteur
        ax1.fill_between(t, i * hauteur, i * hauteur + f, step='post', color=color, label=p)
        ax1.vlines(x=due, ymin=i * hauteur, ymax=i * hauteur + hauteur,
                color=color, linestyle='--', linewidth=2.5)

    ax1.set_yticks([i * hauteur + hauteur / 2 for i in range(n_projets)])
    ax1.set_yticklabels(projets_list)
    ax1.set_ylabel("Projets")
    ax1.set_title("Planning des projets avec créneaux et due date")
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Projets")

    # -------------------
    # Graphique 2 : planning par personne
    # -------------------
    for i, p_data in enumerate(planning_data):
        y = i
        for tache in p_data["taches"]:
            color = project_colors_dict.get(tache["projet"], (0.7, 0.7, 0.7, 0.6))
            ax2.barh(y, tache["t_end"] - tache["t_start"] + 1, left=tache["t_start"], color=color)

    ax2.set_yticks(range(len(S)))
    ax2.set_yticklabels(S)
    ax2.set_xlabel("Temps")
    ax2.set_ylabel("Personnes")
    ax2.set_title("Planning des projets par personne")
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

    # Définir des limites identiques pour les deux axes x
    xmin = min([details['t_min'] for details in projets_faits_details.values()]) - 1
    xmax = max([details['t_max'] for details in projets_faits_details.values()]) + 2
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)

    # Affichage avec Streamlit
    st.pyplot(fig)

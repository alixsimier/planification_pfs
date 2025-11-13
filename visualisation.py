import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st

def plot_planning_complet(projets_faits_details, planning_data_num, S):
    """
    Affiche les créneaux par projet et le planning par personne avec couleurs cohérentes pastel.
    
    projets_faits_details : dict avec t_min, t_max, due_date pour chaque projet
    planning_data_num : matrice personnes x temps avec indices des projets (-1 = vacances)
    S : liste des noms de personnes
    """
    
    projets_list = list(projets_faits_details.keys())
    n_projets = len(projets_list)
    n_personnes = len(S)
    
    # Palette pastel unique
    cmap_base = plt.get_cmap("tab20", n_projets)
    project_colors = [(*cmap_base(i % 20)[:3], 0.6) for i in range(n_projets)]  # RGBA avec alpha=0.6
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8 + n_projets*0.4), constrained_layout=True)
    ax1, ax2 = axes
    
    # -------------------
    # Graphique 1 : créneaux par projet
    # -------------------
    hauteur = 1
    for i, p in enumerate(projets_list):
        details = projets_faits_details[p]
        t_min = details['t_min']
        t_max = details['t_max']
        due = details.get('due_date', t_max)
        
        t = np.arange(t_min, t_max+1)
        f = np.ones_like(t) * hauteur
        color = project_colors[i]
        
        ax1.fill_between(t, i*hauteur, i*hauteur + f, step='post', color=color, label=p)
        ax1.vlines(x=due, ymin=i*hauteur, ymax=i*hauteur + hauteur, color=color, linestyle='--', linewidth=2.5)
    
    ax1.set_yticks([i*hauteur + hauteur/2 for i in range(n_projets)])
    ax1.set_yticklabels(projets_list)
    ax1.set_xlabel("Temps")
    ax1.set_ylabel("Projets")
    ax1.set_title("Planning des projets avec créneaux et due date")
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Projets")
    
    # -------------------
    # Graphique 2 : planning par personne (pastel)
    # -------------------
    new_colors = np.vstack(([1, 1, 1, 1], project_colors))
    new_cmap = mcolors.ListedColormap(new_colors)
    bounds = np.arange(-1, n_projets + 1)
    norm = mcolors.BoundaryNorm(bounds, new_cmap.N)
    
    ax2.imshow(planning_data_num, aspect='auto', cmap=new_cmap, norm=norm, origin='lower')
    
    ax2.set_xlabel("Jour")
    ax2.set_ylabel("Personne")
    ax2.set_yticks(range(n_personnes))
    ax2.set_yticklabels(S)
    ax2.set_title("Planning des projets par personne")
    
    patches = [mpatches.Patch(color=new_cmap(i + 1), label=p) for i, p in enumerate(projets_list)]
    patches.insert(0, mpatches.Patch(color="white", label="Vacances"))
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Projets")
    
    st.pyplot(fig)

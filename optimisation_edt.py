### IMPORTS
from pyomo.environ import (
    ConcreteModel,
    Set,
    Param,
    Var,
    Constraint,
    Objective,
    minimize,
    value,
    RangeSet,
    Binary,
    NonNegativeIntegers,
    maximize,
    ConstraintList,
    SolverFactory, 
    value
)
import copy
import os
import time
from load_instance import load_instance
import matplotlib.pyplot as plt
from pyomo.environ import Param
from rich import print as rprint
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import scipy

### MODÈLE
def solve_model(CONFIG):
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(CONFIG["instance_path"])
    
    ### INITIALISATION 
    m = ConcreteModel("StaffingProjects")
    m.S = Set(initialize=S, ordered=True)
    m.Q = Set(initialize=Q, ordered=True)
    m.P = Set(initialize=P, ordered=True)
    m.T = RangeSet(1, H)

    ### PARAMÈTRES
    m.due = Param(m.P, initialize=due, within=NonNegativeIntegers)

    ### VARIABLES IMMUABLES
    m.SQPT = Set(initialize=[
        (s, q, p, t)
        for s in S for q in Q for p in P for t in m.T
        if eta[s][q] == 1 and mu[p][q] >= 1 and v[s][t] == 1
    ])
    m.lmbda = Var(m.SQPT, within=Binary)
    m.m = Var(m.P, within=Binary)
    m.f = Var(m.P, within=NonNegativeIntegers)
    m.r = Var(m.P, within=NonNegativeIntegers)

    ### CONTRAINTES IMMUABLES
    # Un collaborateur peut être affecté que sur un projet pour une compétence à la fois
    def un_projet_une_competence(m, t, s):
        if v[s][t] == 0:
            return Constraint.Skip
        return sum(m.lmbda[s, q, p, t] for q in m.Q for p in m.P 
                    if (s, q, p, t ) in m.SQPT) <= 1
    m.un_projet_une_competence = Constraint(m.T, m.S, rule=un_projet_une_competence)

    # Un projet est réalisé 
    def projet_fait(m, p, q):
        if mu[p][q] == 0 :
            return Constraint.Skip
        return (sum(m.lmbda[s, q, p, t] for s in m.S for t in m.T 
                    if (s, q, p, t) in m.SQPT) 
                >= m.m[p] * mu[p][q])
    m.projet_fait = Constraint(m.P, m.Q, rule=projet_fait)
    # Date de fin de projet
    def date_fin(m, s, p, q, t):
        if (s, q, p, t) not in m.SQPT:
            return Constraint.Skip
        return m.f[p] >= t * m.lmbda[s, q, p, t]
    m.date_fin = Constraint(m.S, m.P, m.Q, m.T, rule=date_fin)
    # Défintion du retard pour un projet
    def projet_retard(m, p):
        return m.r[p] >= m.f[p] - due[p]
    m.retard = Constraint(m.P, rule=projet_retard)

    m.expr_profit = sum(gain[p] * m.m[p] - pen[p] * m.r[p] for p in m.P)

    ### CONTRAINTES OPTIONNELLES

    objs = [CONFIG["obj_principale"], *CONFIG["obj_secondaires"].keys()]
    
    if "duree" in objs :
        m.d = Var(m.P, within=NonNegativeIntegers)
        # Date de début de projet
        def date_debut(m, s, q, p, t):
            if (s, q, p, t) not in m.SQPT:
                return Constraint.Skip
            return m.d[p] <= t + H*(1- m.lmbda[s, q, p, t])
        m.date_debut = Constraint(m.S, m.Q, m.P, m.T, rule=date_debut)
        # Durée positive
        def duree_pos(m, p):
            return m.d[p] <= m.f[p]
        m.duree_pos = Constraint(m.P, rule=duree_pos)
        m.expr_duree = sum([m.f[p] - m.d[p] for p in m.P])
    
    if "personne" in objs:
        m.t = Var(m.S, m.P, within=Binary)
        # Si la personne S travaille sur le projet P
        def projet_personne(m, s, p):
            return len(m.P) * H * m.t[s,p] >= sum(m.lmbda[s,q,p,t] for q in m.Q for t in m.T if (s, q, p, t ) in m.SQPT)
        m.projet_personne = Constraint(m.S, m.P, rule=projet_personne)
        m.expr_personne = sum(m.t[s,p] for s in m.S for p in m.P) / len(m.S)

    if "projet" in objs:
        m.expr_projet = sum(m.m[p] for p in m.P)

    if "retard" in objs:
        m.R = Var(m.P, within=Binary)
        # Si un projet est rendu en retard
        def projet_retard_bin(m, p):
            return m.R[p] >= m.r[p]/H
        m.projet_retard_bin = Constraint(m.P, rule=projet_retard_bin)
        m.expr_retard = sum(m.R[p] for p in m.P)

    ### OBJECTIFS

    if CONFIG["obj_principale"] == "duree":
        m.OBJ = Objective(expr=m.expr_duree, sense=minimize)
    elif CONFIG["obj_principale"] == "profit":
        m.OBJ = Objective(expr=m.expr_profit, sense=maximize)
    elif CONFIG["obj_principale"] == "projet":
        m.OBJ = Objective(expr=m.expr_projet, sense=maximize)
    elif CONFIG["obj_principale"] == "personne":
        m.OBJ = Objective(expr=m.expr_personne, sense=minimize)
    elif CONFIG["obj_principale"] == "retard":
        m.OBJ = Objective(expr=m.expr_retard, sense=minimize)

    for obje, eps in CONFIG["obj_secondaires"].items():
        if obje == "duree":
            m.c_duree = Constraint(m.P, rule=lambda m, p: m.f[p] - m.d[p] <= eps)
            # m.c_duree = Constraint(expr=m.expr_duree <= eps)
        elif obje == "profit":
            m.c_profit = Constraint(expr=m.expr_profit >= eps)
        elif obje == "projet":
            m.c_projet = Constraint(expr=m.expr_projet >= eps)
        elif obje == "personne":
            m.c_personne = Constraint(expr=m.expr_personne <= eps)
        elif obje == "retard":
            m.c_retard = Constraint(expr=m.expr_retard <= eps)

    ### RESOLUTION
    opt = SolverFactory(CONFIG["solver"])
    opt.options["TimeLimit"] = 10
    results = opt.solve(m, tee=CONFIG["tee"])
    return m, results

def calculs(m, results):
    nb_projets_par_personne = {s: 0 for s in m.S}
    for s in m.S:
        projets_actifs = set()
        for (s2, q, p, t) in m.SQPT:
            if s2 == s and value(m.lmbda[s2, q, p, t]) > 0.5:
                projets_actifs.add(p)
        nb_projets_par_personne[s] = len(projets_actifs)
    nb_projet_moyen = sum(nb_projets_par_personne.values()) / len(m.S)
    temps_par_projet = {}
    for (s, q, p, t) in m.SQPT:
        if value(m.lmbda[s, q, p, t]) > 0.5:
            if p not in temps_par_projet:
                temps_par_projet[p] = {'t_min': t, 't_max': t}
            else:
                temps_par_projet[p]['t_min'] = min(temps_par_projet[p]['t_min'], t)
                temps_par_projet[p]['t_max'] = max(temps_par_projet[p]['t_max'], t)
    durees = []
    projets_faits_details = {}
    for p in temps_par_projet:
        t_min = temps_par_projet[p]['t_min']
        t_max = temps_par_projet[p]['t_max']
        duree = t_max - t_min + 1
        temps_par_projet[p]['durée'] = duree
        durees.append(duree)
        projets_faits_details[p] = {
            't_min': t_min,
            't_max': t_max,
            'durée': duree,
            'due_date': value(m.due[p])  # m.due[p] dépend seulement de p
        }
    duree_moyenne = sum(durees) / len(durees) if durees else 0
    projet = sum([m.m[p].value for p in m.P])
    retard = 0
    for p in m.P:
        if value(m.m[p]) > 0:
            if value(m.f[p]) > m.due[p]:
                retard += 1
    rprint(temps_par_projet)
    return value(m.expr_profit), nb_projet_moyen, duree_moyenne, projet, retard, projets_faits_details

# def debug_dates(m, H):
#     print("=== DEBUG DéBUT / FIN ===")
#     for p in m.P:
#         print(f"\nProjet {p} : d = {value(m.d[p])}, f = {value(m.f[p])}")
#         for (s, q, pp, t) in m.SQPT:
#             if pp != p:
#                 continue
#             lam = value(m.lmbda[s, q, pp, t])
#             if lam > 0.5:  # on ne regarde que les lambda = 1
#                 lhs = value(m.d[pp])
#                 rhs = t + (H - t) * (1 - lam)
#                 print(f"  s={s}, q={q}, t={t}, λ={lam:.3f}  -->  d[p]={lhs}  <= RHS={rhs}")

def pareto2(CONFIG, obj_princ, contr_name, objectif, contrainte1):
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(CONFIG["instance_path"])
    pareto_points = []

    # Définir les valeurs possibles pour la contrainte secondaire
    if contr_name in ["duree", "retard"]:
        eps_values = list(range(H, 0, -1))
    elif contr_name == "personne":
        eps_values = list(range(len(P), 0, -1))
    elif contr_name == "projet":
        eps_values = list(range(1, len(P)+1))
    else:
        raise ValueError(f"Contrôle secondaire inconnu : {contr_name}")

    for eps in eps_values:
        config_eps = copy.deepcopy(CONFIG)
        config_eps["obj_secondaires"] = {contr_name: eps}

        m, results = solve_model(config_eps)

        if str(results.solver.termination_condition) != "optimal":
            rprint(f"Pas de solution pour contrainte : {contrainte1}={eps}")
            continue

        profit, personne, duree, projet, retard, _ = calculs(m, results)
        valeurs = {
            "profit": profit,
            "personne": personne,
            "durée": duree,
            "projet": projet,
            "retard": retard
        }

        pareto_points.append((
            valeurs.get(obj_princ, 0),
            valeurs.get(contr_name, 0)
        ))

    if not pareto_points:
        rprint("Aucun point de Pareto valide trouvé.")
        return []

    # Tracé 2D
    princ_vals, contr_vals = zip(*pareto_points)
    plt.figure(figsize=(8,6))
    sc = plt.plot(contr_vals, princ_vals, '-o', color='blue', markersize=6, linewidth=1.5)
    plt.ylabel(objectif)
    plt.xlabel(contrainte1)
    plt.title(f"Front de Pareto: {objectif} vs {contrainte1}")
    plt.grid(True)
    plt.show()

    return pareto_points


def pareto3(CONFIG, obj_princ, contrainte1, contrainte2):
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(CONFIG["instance_path"])
    pareto_points = []
    CONFIG['obj_principale'] = obj_princ

    # Définir les valeurs possibles pour chaque contrainte
    def eps_values(contr, H, P):
        if contr in ["duree", "retard"]:
            return list(range(H, 0, -1))
        elif contr == "personne":
            return list(range(len(P), 0, -1))
        elif contr == "projet":
            return list(range(1, len(P)+1))
        else:
            raise ValueError(f"Contrôle secondaire inconnu : {contr}")

    eps1_list = eps_values(contrainte1, H, P)
    eps2_list = eps_values(contrainte2, H, P)

    for eps1 in eps1_list:
        for eps2 in eps2_list:
            config_eps = copy.deepcopy(CONFIG)
            config_eps["obj_secondaires"] = {contrainte1: eps1, contrainte2: eps2}
            m, results = solve_model(config_eps)

            if str(results.solver.termination_condition) != "optimal":
                rprint(f"Pas de solution pour {contrainte1}={eps1}, {contrainte2}={eps2}")
                continue

            profit, personne, duree, projet, retard, _ = calculs(m, results)
            valeurs = {
                "profit": profit,
                "personne": personne,
                "durée": duree,
                "projet": projet,
                "retard": retard
            }

            pareto_points.append((
                valeurs.get(obj_princ, 0),
                valeurs.get(contrainte1, 0),
                valeurs.get(contrainte2, 0)
            ))

    if not pareto_points:
        rprint("Aucun point de Pareto valide trouvé.")
        return []

    # Tracer les points 3D
    princ, contr1_vals, contr2_vals = zip(*pareto_points)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(princ, contr1_vals, contr2_vals, c=princ, cmap='viridis', marker='o')
    ax.set_xlabel(obj_princ)
    ax.set_ylabel(contrainte1)
    ax.set_zlabel(contrainte2)
    ax.set_title(f"Points de Pareto: {obj_princ} vs {contrainte1} vs {contrainte2}")
    plt.show()

    return pareto_points

def pareto_surface_3D(points, objectif, contrainte1, contrainte2):
    if not points:
        rprint("Aucun point à afficher pour la surface.")
        return

    obj_princ, contr1_vals, contr2_vals = zip(*points)
    obj_princ = np.array(obj_princ)
    contr1_vals = np.array(contr1_vals)
    contr2_vals = np.array(contr2_vals)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    # Ajouter un petit bruit pour éviter les doublons exacts
    eps = 1e-6
    obj_princ_j = obj_princ + np.random.normal(0, eps, len(obj_princ))
    contr1_j = contr1_vals + np.random.normal(0, eps, len(contr1_vals))

    # Grille pour la surface
    xi = np.linspace(min(obj_princ_j), max(obj_princ_j), 50)
    yi = np.linspace(min(contr1_j), max(contr1_j), 50)
    XI, YI = np.meshgrid(xi, yi)

    ZI = griddata((obj_princ_j, contr1_j), contr2_vals, (XI, YI), method='linear')

    surf = ax.plot_surface(XI, YI, ZI, cmap=cm.viridis, edgecolor='none', alpha=0.8)
    ax.set_xlabel(objectif)
    ax.set_ylabel(contrainte1)
    ax.set_zlabel(contrainte2)
    ax.set_title(f"Surface de Pareto: {objectif} vs {contrainte1} vs {contrainte2}")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=objectif)
    plt.show()

# if __name__ == "__main__":
#     CONFIG = {
#         "instance_path" : "instances/medium_instance.json",
#         "obj_principale" : "durée", 
#         "solver":"gurobi",
#         "tee": False,
#         # "obj_secondaires" : {"projets" : 3}
#         "obj_secondaires" : {"projets":2, "profit":30}
#         # "obj_secondaires" : {}
#     }
#     m, res = solve_model(CONFIG)
#     rprint(res.solver.termination_condition)
#     prof, personne, duree, projet = calculs(m, res)
#     print(prof)
#     print(personne)
#     print(duree)
#     print(projet)
#     for p in m.P:
#         if m.m[p].value > 0:
#             print(p, m.f[p].value, m.d[p].value)
#     debug_dates(m, 22)

def run_benchmarks_folder(folder_path, CONFIG_template):
    all_times = [] 
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".json"):
            continue
        instance_path = os.path.join(folder_path, file_name)
        print(f"\n=== Running benchmark for {file_name} ===")
        CONFIG = CONFIG_template.copy()
        CONFIG["instance_path"] = instance_path
        start_time = time.time()
        try :
            m, results = solve_model(CONFIG)
            end_time = time.time()
            elapsed = end_time - start_time
            all_times.append(elapsed)
            print(f"Time for {file_name}: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"Exception as {e}")
    plt.figure(figsize=(6,4))
    plt.hist(all_times, color='skyblue', bins=30, edgecolor='black')
    plt.title(f"Histogramme des temps de calcul")
    plt.ylabel("Nombre d'instances")
    plt.xlabel("Temps en secondes")
    plt.grid(axis='y')
    plt.show()
    
    return all_times

if __name__ == "__main__":
    
    CONFIG = {
        "instance_path" : "instances/generation_instance.json",
        "obj_principale" : "profit", 
        "solver":"gurobi",
        "tee": False,
        "obj_secondaires" : {}
    }
    # res = run_benchmarks_folder("instances/generated_instances", CONFIG)
    # m, res = solve_model(CONFIG)
    # c = calculs(m, res)
    # rprint(c)
    # # obj_princ = "profit"
    # # contrainte1 = "duree"
    # # contrainte2 = "personne"
    pareto_points = pareto3(CONFIG, "profit", "duree", "personne")
    # # pareto_points = pareto3(CONFIG, obj_princ, contrainte1, contrainte2)
    # # print(pareto_points)
    pareto_surface_3D(pareto_points, "Profit", "Nombre de projets en retard", "Nombre moyen de projets par personne")

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
    for p in temps_par_projet:
        t_min = temps_par_projet[p]['t_min']
        t_max = temps_par_projet[p]['t_max']
        duree = t_max - t_min + 1
        temps_par_projet[p] = duree
        durees.append(duree)
    duree_moyenne = sum(durees) / len(durees) if durees else 0
    projet = sum([m.m[p].value for p in m.P])
    retard = 0
    for p in m.P:
        if value(m.m[p]) > 0:
            if value(m.f[p]) > m.due[p]:
                retard += 1
    return value(m.expr_profit), nb_projet_moyen, duree_moyenne, projet, retard

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

def pareto2(CONFIG, obj_princ, contr1, objectif, contrainte1):
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(CONFIG["instance_path"])
    pareto_points = []
    CONFIG['obj_principale'] = obj_princ
    if contr1 == "duree" or contr1 == "retard":
        i, j, k = H, -1, -1
    if contr1 == "personne" :
        i, j, k = len(P), -1, -1
    if contr1 == "projet" :
        i, j, k = 1, len(P), -1
    for eps in range(i,j,k):
        config_eps = CONFIG.copy()
        config_eps["obj_secondaires"][contr1] = eps
        m, results = solve_model(config_eps)
        if str(results.solver.termination_condition) != "optimal":
            rprint(f"Pas de solution pour contrainte : {contrainte1}={eps}")
            continue
        profit, personne, duree, projet, retard = calculs(m, results)
        rprint("Profits réalisés ", profit)
        rprint("Nombre moyen de projets par personne ", personne)
        rprint("Durée moyenne des projets ", duree)
        rprint("Nombre de projets réalisés ", projet)
        rprint("Nombre de projets en retard ", retard)

        valeurs = {
            "profit": profit,
            "personne": personne,
            "durée": duree,
            "projet": projet,
            "retard": retard
        }

        obj_princ = CONFIG["obj_principale"]
        contr = CONFIG["obj_secondaires"][contr1]

        pareto_points.append((
            valeurs.get(obj_princ, 0),
            valeurs.get(contr, 0),
        ))

    if not pareto_points:
        rprint("Aucun point de Pareto valide trouvé.")
        return []

    # Décomposer les points
    princ, contr1 = zip(*pareto_points)

    # Tracer la surface 3D
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(princ, contr1, c=princ, cmap='viridis', marker='o')
    ax.set_xlabel(obj_princ)
    ax.set_ylabel(contrainte1)
    ax.set_title(f"Front de Pareto: {obj_princ} vs {contrainte1}")
    plt.colorbar(sc, label={obj_princ})
    plt.show()

    return pareto_points


def pareto3(CONFIG, obj_princ, contrainte1, contrainte2):
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(CONFIG["instance_path"])
    pareto_points = []
    CONFIG['obj_principale'] = obj_princ
    if contrainte1 == "duree" or contrainte1 == "retard":
        i, j, k = H, -1, -1
    if contrainte1 == "personne" :
        i, j, k = len(P), -1, -1
    if contrainte1 == "projet" :
        i, j, k = 1, len(P), -1
    if contrainte2 == "duree" or contrainte2 == "retard":
        l, o, n = H, -1, -1
    if contrainte2 == "personne" :
        l, o, n = len(P), -1, -1
    if contrainte2 == "projet" :
        l, o, n = 1, len(P), -1
    for eps in range(i,j,k):
        config_eps = CONFIG.copy()
        config_eps["obj_secondaires"][contrainte1] = eps
        for e in range(l,o,n):
            config_eps["obj_secondaires"][contrainte2] = e
            m, results = solve_model(config_eps)
            if str(results.solver.termination_condition) != "optimal":
                rprint(f"Pas de solution pour contrainte : {contrainte1}={eps}, et contrainte : {contrainte2}={e}")
                continue
            profit, personne, duree, projet, retard = calculs(m, results)
            rprint("Profits réalisés ", profit)
            rprint("Nombre moyen de projets par personne ", personne)
            rprint("Durée moyenne des projets ", duree)
            rprint("Nombre de projets réalisés ", projet)
            rprint("Nombre de projets en retard ", retard)

            valeurs = {
                "profit": profit,
                "personne": personne,
                "durée": duree,
                "projet": projet,
                "retard": retard
            }

            obj_princ = CONFIG["obj_principale"]
            contrainte1, contrainte2 = list(CONFIG["obj_secondaires"].keys())[:2]

            pareto_points.append((
                valeurs.get(obj_princ, 0),
                valeurs.get(contrainte1, 0),
                valeurs.get(contrainte2, 0)
            ))

    if not pareto_points:
        rprint("Aucun point de Pareto valide trouvé.")
        return []

    # Décomposer les points
    princ, contr1, contr2 = zip(*pareto_points)

    # Tracer la surface 3D
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(princ, contr1, contr2, c=princ, cmap='viridis', marker='o')
    ax.set_xlabel(obj_princ)
    ax.set_ylabel(contrainte1)
    ax.set_zlabel(contrainte2)
    ax.set_title(f"Surface de Pareto: {obj_princ} vs {contrainte1} vs {contrainte2}")
    plt.colorbar(sc, label={obj_princ})
    plt.show()

    return pareto_points

def pareto_surface_3D(points, objectif, contrainte1, contrainte2):
    obj_princ, contr1_vals, contr2_vals = zip(*points)
    obj_princ = np.array(obj_princ)
    contr1_vals = np.array(contr1_vals)
    contr2_vals = np.array(contr2_vals)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    try :
        xi = np.linspace(min(obj_princ), max(obj_princ), 50)
        yi = np.linspace(min(contr1_vals), max(contr1_vals), 50)
        XI, YI = np.meshgrid(xi, yi)

        ZI = griddata(
            (obj_princ, contr1_vals), contr2_vals,
            (XI, YI), method='linear'
        )
        surf = ax.plot_surface(XI, YI, ZI, cmap=cm.viridis, edgecolor='none', alpha=0.8)
    except :
        eps = 1e-5
        obj_princ = np.array(obj_princ) + np.random.normal(0, eps, len(obj_princ))
        contr1_vals = np.array(contr1_vals) + np.random.normal(0, eps, len(contr1_vals))
        contr2_vals = np.array(contr2_vals) + np.random.normal(0, eps, len(contr2_vals))
        xi = np.linspace(min(obj_princ), max(obj_princ), 50)
        yi = np.linspace(min(contr1_vals), max(contr1_vals), 50)
        XI, YI = np.meshgrid(xi, yi)

        ZI = griddata(
            (obj_princ, contr1_vals), contr2_vals,
            (XI, YI), method='linear'
        )
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

if __name__ == "__main__":
    
    CONFIG = {
        "instance_path" : "instances/toy_instance_v.json",
        "obj_principale" : "profit", 
        "solver":"gurobi",
        "tee": False,
        "obj_secondaires" : {}
    }
    m, res = solve_model(CONFIG)
    c = calculs(m, res)
    rprint(c)
    obj_princ = "profit"
    contrainte1 = "duree"
    contrainte2 = "personne"
    pareto_points = pareto2(CONFIG, obj_princ, "duree", "Profit", "Durée moyenne d'un projet")
    # pareto_points = pareto3(CONFIG, obj_princ, contrainte1, contrainte2)
    # print(pareto_points)
    # pareto_surface_3D(pareto_points, "Profit", "Durée moyenne d'un projet", "Nombre moyen de projets par personne")

from ast import Expression
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
    SolverFactory
)
from load_instance import load_instance
import matplotlib.pyplot as plt
from pyomo.environ import Param
from rich import print as rprint

def solve_model(CONFIG):
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(CONFIG["instance_path"])
    
    ### INITIALISATION 
    m = ConcreteModel("StaffingProjects")
    m.S = Set(initialize=S, ordered=True)
    m.Q = Set(initialize=Q, ordered=True)
    m.P = Set(initialize=P, ordered=True)
    m.T = RangeSet(1, H)

    ### VARIABLES
    m.SQPT = Set(initialize=[
        (s, q, p, t)
        for s in S for q in Q for p in P for t in m.T
        if eta[s][q] == 1 and mu[p][q] >= 1 and v[s][t] == 1
    ])
    m.lmbda = Var(m.SQPT, within=Binary)
    m.m = Var(m.P, within=Binary)
    m.f = Var(m.P, within=NonNegativeIntegers)
    m.r = Var(m.P, within=NonNegativeIntegers)
    m.t = Var(m.S, m.P, within=Binary)
    m.d = Var(m.P, within=NonNegativeIntegers)

    ### CONTRAINTES
    # Si la personne S travaille sur le projet P
    def projet_personne(m, s, p):
        return m.t[s,p] >= sum(m.lmbda[s,q,p,t] for q in m.Q for t in m.T if (s, q, p, t ) in m.SQPT) / H
    m.projet_personne = Constraint(m.S, m.P, rule=projet_personne)

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

    def durée_pos(m, p):
        return m.d[p] <= m.f[p]
    m.durée_pos = Constraint(m.P, rule=durée_pos)
    
    # Date de début de projet
    def date_début(m, s, p, q, t):
        if (s, q, p, t) not in m.SQPT:
            return Constraint.Skip
        return m.d[p] <= t + ( max(m.T)-t)*(1- m.lmbda[s, q, p, t])
    m.date_début = Constraint(m.S, m.P, m.Q, m.T, rule=date_début)

    ### OBJECTIF

    # profit
    m.expr_profit = sum(gain[p] * m.m[p] - pen[p] * m.r[p] for p in m.P)
    # m.OBJ = Objective(expr=expr_profit, sense=maximize)

    # projets faits
    m.expr_projets = sum(m.m[p] for p in m.P)
    # m.OBJ = Objective(expr=expr_projets, sense=maximize)

    # durée projets
    m.expr_duree = sum(m.f[p] - m.d[p] for p in m.P)
    # m.OBJ = Objective(expr=expr_projets, sense=maximize)

    # compacité du nombre de projets par personnes
    m.expr_compacite = sum(m.t[s,p] for s in m.S for p in m.P) / len(m.S)


    if CONFIG["obj_principale"] == "durée":
        m.OBJ = Objective(expr=m.expr_duree, sense=minimize)
    elif CONFIG["obj_principale"] == "profit":
        m.OBJ = Objective(expr=m.expr_profit, sense=maximize)
    elif CONFIG["obj_principale"] == "projets":
        m.OBJ = Objective(expr=m.expr_projets, sense=maximize)
    elif CONFIG["obj_principale"] == "compacité":
        m.OBJ = Objective(expr=m.expr_compacite, sense=minimize)

    m.epsilon_constraints = ConstraintList()

    for obje, eps in CONFIG["obj_secondaires"].items():
        if obje == "durée":
            m.epsilon_constraints.add(expr=m.expr_duree <= eps)
        elif obje == "profit":
            m.epsilon_constraints.add(expr=m.expr_profit >= eps)
        elif obje == "projets":
            m.epsilon_constraints.add(expr=m.expr_projets >= eps)
        elif obje == "compacité":
            m.epsilon_constraints.add(expr=m.expr_compacite <= eps)

    # Résolution
    opt = SolverFactory(CONFIG["solver"])
    results = opt.solve(m, tee=CONFIG["tee"])
    return m, results

def pareto_profit_compacite(CONFIG):
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(CONFIG["instance_path"])
    
    pareto_points = []

    for eps in range(len(Q), -1, -1):
        config_eps = CONFIG.copy()
        rprint(eps)
        config_eps["obj_secondaires"] = {"compacité": eps}

        m, results = solve_model(config_eps)

        print(results)

        # Calculer profit et compacité pour ce modèle
        profit = value(m.OBJ)
        compacite = value(m.expr_compacite)

        rprint(profit)
        rprint(compacite)
        pareto_points.append((profit, compacite))

    # Tracer la surface de Pareto
    profits, compacites = zip(*pareto_points)
    plt.figure(figsize=(7,5))
    plt.plot(compacites, profits, marker='o')
    plt.xlabel("Compacité (nombre de projets moyen par personne)")
    plt.ylabel("Profit")
    plt.title("Front de Pareto: Profit vs Compacité")
    plt.grid(True)
    plt.show()

    return pareto_points


if __name__ == "__main__":
    CONFIG = {
        "instance_path" : "instances/medium_instance.json",
        "obj_principale" : "durée", 
        "solver":"gurobi",
        "tee": False,  # True si vous voulez voir le log solver
        "obj_secondaires" : {"profit" : 40}  # sera mis à jour dans la boucle epsilon
    }
    m, res = solve_model(CONFIG)
    print(value(m.expr_profit))
    print(value(m.expr_duree))
    print(value(m.expr_projets))
    print(value(m.expr_compacite))

    # points = pareto_profit_compacite(CONFIG)


# if __name__ == "__main__":
    
#     CONFIG = {
#         "instance_path" : "instances/medium_instance.json",
#         "obj_principale" : "profit", 
#         "solver":"cbc",
#         "tee": True,
#         "obj_secondaires" : {"durée": 30, "compacité": 15}
#     }
#     results = solve_model(CONFIG)
#     print(results)

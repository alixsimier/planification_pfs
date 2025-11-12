from ast import Expression
from pyomo.environ import (
ConcreteModel, Set, RangeSet, Param, Var, Binary, NonNegativeIntegers,
Objective, Constraint, maximize, value, SolverFactory, Expression
)
from load_instance import load_instance
import matplotlib.pyplot as plt
from pyomo.environ import Param


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

    # Date de début de projet
    def date_début(m, s, p, q, t):
        if (s, q, p, t) not in m.SQPT:
            return Constraint.Skip
        return m.d[p] <= t +(H-t)*(1- m.lmbda[s, q, p, t])
    m.date_début = Constraint(m.S, m.P, m.Q, m.T, rule=date_début)

    ### OBJECTIF

    # profit
    expr_profit = sum(gain[p] * m.m[p] - pen[p] * m.r[p] for p in m.P)
    # m.OBJ = Objective(expr=expr_profit, sense=maximize)

    # projets faits
    expr_projets = sum(m.m[p] for p in m.P)
    # m.OBJ = Objective(expr=expr_projets, sense=maximize)

    # durée projets
    expr_durée = -sum(m.f[p] - m.d[p] for p in m.P)
    # m.OBJ = Objective(expr=expr_projets, sense=maximize)

    # compacité du nombre de projets par personnes
    expr_compacité = -sum(m.t[s,p] for s in m.S for p in m.P) / len(m.S)


    # if CONFIG["objective"] == "projets":
    #     m.OBJ = Objective(expr=expr_projets, sense=maximize)
    if CONFIG["objective"] == "durée":
        m.OBJ = Objective(expr=expr_durée, sense=maximize)
    if CONFIG["objective"] == "profit":
        m.OBJ = Objective(expr=expr_profit, sense=maximize)
    if CONFIG["objective"] == "compacité":
        m.OBJ = Objective(expr=expr_compacité, sense=maximize)
   
    opt = SolverFactory(CONFIG["solver"])
    results = opt.solve(m, tee=True)
    return m, results

if __name__ == "__main__":
    
    CONFIG = {
        "instance_path" : "instances/medium_instance.json",
        "objective":"projets", # profit, projets, multi
        "w_profit": 0.8, # si multi objectif
        "max_projet_par_pers": 5, # si multi objectif
        "solver":"gurobi",
        "tee": True
    }
    results = solve_model(CONFIG)
    print(results)

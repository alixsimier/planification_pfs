from pyomo.environ import (
    ConcreteModel, Set, RangeSet, Param, Var, Binary, NonNegativeIntegers,
    Objective, Constraint, maximize, value, SolverFactory
)
from load_instance import load_instance

INSTANCE_PATH = "instances/toy_instance.json"  
OBJECTIVE     = "profit"
W_PROFIT      = 1.0
W_NPROJ       = 0.0
W_ONTIME      = 0.0
SOLVER_NAME   = "glpk"
TEE           = True
TIMELIMIT     = None


if __name__ == "__main__":
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(INSTANCE_PATH)
    objective = "profit"

    ### INITIALISATION 
    m = ConcreteModel("StaffingProjects")
    m.S = Set(initialize=S, ordered=True)
    m.Q = Set(initialize=Q, ordered=True)
    m.P = Set(initialize=P, ordered=True)
    m.T = RangeSet(1, H)

    ### VARIABLES
    index = [(s, q, p, t) for s in S for q in Q for p in P for t in m.T if eta[s][q] == 1 and mu[p][q] >= 1 and v[s][t] == 1]
    m.lmbda = Var(index , within=Binary)
    m.z = Var(m.P, m.T, within=Binary)
    m.m = Var(m.P, within=Binary)
    m.u     = Var(m.P, m.T, within=Binary)

    ### PARAMÈTRES
    m.g   = Param(m.P, initialize=lambda m,p: int(gain[p]), within=NonNegativeIntegers)
    m.d   = Param(m.P, initialize=lambda m,p: int(due[p]), within=NonNegativeIntegers)
    m.p   = Param(m.P, initialize=lambda m,p: int(pen[p]), within=NonNegativeIntegers)
    m.eta = Param(m.S, m.Q, initialize=lambda m,s,q: int(eta[s][q]), within=Binary)                                                 # η_{s,q}
    m.v   = Param(m.S, m.T, initialize=lambda m,s,t: int(v[s][t]), within=Binary)
    m.mu  = Param(m.P, m.Q, initialize=lambda m,p,q: int(mu[p][q]), within=NonNegativeIntegers)                                     # μ_{p,q}

    ### OBJECTIF

    # profit
    expr_retard = sum(m.p[p] * m.u[p,t] for p in m.P for t in m.T if t >= value(m.d[p]))
    expr_profit = sum(m.g[p] * m.m[p] for p in m.P) - expr_retard
    m.OBJ = Objective(expr=expr_profit, sense=maximize)

    # projets faits
    expr_projets = sum(m.m[p] for p in m.P)
    # m.OBJ = Objective(expr=expr_projets, sense=maximize)


    ### CONTRAINTES
    # Un collaborateur peut être affecté que sur un projet pour une compétence à la fois
    def un_projet_une_competence(m, t, s):
        expr = []
        for p in m.P:
            for q in m.Q:
                if eta[s][q] == 1 and  m.mu[p,q] >=  1 and m.v[s,t] == 1 :
                    expr.append(m.lmbda[s,q,p,t])
        if not expr:
            return Constraint.Skip
        return sum(expr) <= 1
    m.un_projet_une_competence = Constraint(m.T, m.S, rule=un_projet_une_competence)

    # Un projet est réalisé que si tous les jours de travails sont faits
    def projet_fait(m, p, q):
        expr = []
        for s in m.S:
            for t in m.T:
                if m.eta[s,q] == 1 and  m.mu[p,q]>= 1 and m.v[s,t] == 1 :
                    expr.append(m.lmbda[s,q,p,t])
        if not expr:
            return Constraint.Skip
        return sum(expr) >= m.m[p]*m.mu[p,q]
    m.projet_fait = Constraint(m.P, m.Q, rule=projet_fait)
    def projet_fait_T(m, p, q, T):
        expr = []
        for s in m.S:
            for t in m.T:
                if m.eta[s,q] == 1 and  m.mu[p,q]>= 1 and m.v[s,t] == 1 and t <= T :
                    expr.append(m.lmbda[s,q,p,t])
        if not expr:
            return Constraint.Skip
        return sum(expr)  >= m.z[p, t]* m.mu[p,q]
    m.projet_fait_T = Constraint(m.P, m.Q, m.T, rule=projet_fait_T)

    # # Contrainte de monotonie sur z(p,t)
    # def monotonie_z(m, p, t):
    #     if t < H :
    #         return m.z[p, t+1] >= m.z[p, t]
    #     else:
    #         return Constraint.Skip
    # m.monotonie_z = Constraint(m.P, m.T, rule=monotonie_z)
    # Linéariser u_p,t
    def rule1(m, p, t):
        return m.u[p,t] <= m.m[p]
    m.rule1 = Constraint(m.P, m.T, rule=rule1)
    def rule2(m, p, t):
        return m.u[p,t] <= 1 - m.z[p, t]
    m.rule2 = Constraint(m.P, m.T, rule=rule2)
    def rule3(m, p, t):
        return m.u[p,t] >= m.m[p] - m.z[p, t]
    m.rule3 = Constraint(m.P, m.T, rule=rule3)

    ### RÉSOLUTION
    opt = SolverFactory("glpk")
    results = opt.solve(m, tee=True)
    print("ok", results)
    print('\n\n\n')
    print(results.solver.status)
    print('\n\n\n')
    print(results.solver.termination_condition)
    obj_val = value(m.OBJ)
    print("Valeur de l'objectif :", obj_val)
    print("Projets réalisés :")
    for p in m.P:
        if value(m.m[p]) > 0.5:  # binaire, donc >0.5 signifie 1
            print(f"Projet {p} est réalisé (m[p]={value(m.m[p])})")
    print("Affectations du personnel :")
    for s, q, p, t in m.lmbda:
        if value(m.lmbda[s,q,p,t]) > 0.5:
            print(f"Personne {s} travaille sur projet {p}, compétence {q}, jour {t}")
    for p in m.P:
        for t in m.T:
            if value(m.z[p,t]) < 0.5 :
                print(f"Le projet {p} n'est pas réalisé à l'instant {t}")
            else :
                print(f"Le projet {p} est réalisé à l'instant {t}")

# Paramètres

INSTANCE_PATH = "instances/medium_instance.json"  
OBJECTIVE     = "profit"                            # "profit" | "nb_projects" | "on_time" | "combo"
W_PROFIT      = 1.0                                  # poids (si OBJECTIVE="combo")
W_NPROJ       = 0.0
W_ONTIME      = 0.0
SOLVER_NAME   = "cbc"
TEE          = True
TIMELIMIT     = None


# Imports nécessaires

import json
from collections import defaultdict
from pyomo.environ import (
    ConcreteModel, Set, RangeSet, Param, Var, Binary, NonNegativeIntegers,
    Objective, Constraint, maximize, value, SolverFactory
)
from load_instance import load_instance



# Modèle 

def build_model(H, I, J, K, eta, v, mu, gain, due, pen,
                objective="profit", w_profit=1.0, w_nproj=0.0, w_ontime=0.0):
    m = ConcreteModel("StaffingProjects")

    # Ensembles
    m.I = Set(initialize=I, ordered=True)
    m.J = Set(initialize=J, ordered=True)
    m.K = Set(initialize=K, ordered=True)
    m.T = RangeSet(1, H)

    # Paramètres
    m.eta = Param(m.I, m.J, initialize=lambda m,i,j: int(eta[i][j]) if j in eta[i] else 0, within=Binary, default=0)
    m.v   = Param(m.I, m.T, initialize=lambda m,i,t: int(v[i][t]), within=Binary, default=1)
    m.mu  = Param(m.K, m.J, initialize=lambda m,k,j: int(mu[k][j]) if j in mu[k] else 0,
                  within=NonNegativeIntegers, default=0)
    m.g   = Param(m.K, initialize=lambda m,k: int(gain[k]), within=NonNegativeIntegers, default=0)
    m.d   = Param(m.K, initialize=lambda m,k: max(1, min(H, int(due[k]))), within=NonNegativeIntegers)
    m.p   = Param(m.K, initialize=lambda m,k: int(pen[k]), within=NonNegativeIntegers, default=0)

    # Variables
    m.x     = Var(m.I, m.J, m.K, m.T, within=Binary)  # affectations
    m.z     = Var(m.K, m.T, within=Binary)            # complétion cumulée
    m.w     = Var(m.K, m.T, within=Binary)            # complétion exacte au jour t
    m.mdone = Var(m.K, within=Binary)                 # projet réalisé

    # Contraintes
    def _qual_rule(m, i, j, k, t):
        return m.x[i,j,k,t] <= m.eta[i,j]
    m.Qualif = Constraint(m.I, m.J, m.K, m.T, rule=_qual_rule)

    def _one_task_per_day_rule(m, i, t):
        return sum(m.x[i,j,k,t] for j in m.J for k in m.K) <= m.v[i,t]
    m.OneTaskPerDay = Constraint(m.I, m.T, rule=_one_task_per_day_rule)

    def _total_cover_rule(m, k, j):
        req = m.mu[k,j]
        if value(req) == 0:
            return Constraint.Skip
        return sum(m.x[i,j,k,t] for i in m.I for t in m.T) >= req * m.mdone[k]
    m.TotalCover = Constraint(m.K, m.J, rule=_total_cover_rule)

    def _cum_cover_rule(m, k, j, t):
        req = m.mu[k,j]
        if value(req) == 0:
            return Constraint.Skip
        return sum(m.x[i,j,k,tt] for i in m.I for tt in m.T if tt <= t) >= req * m.z[k,t]
    m.CumCover = Constraint(m.K, m.J, m.T, rule=_cum_cover_rule)

    def _z_mono_rule(m, k, t):
        if t == 1:
            return Constraint.Skip
        return m.z[k,t-1] <= m.z[k,t]
    m.ZMonotone = Constraint(m.K, m.T, rule=_z_mono_rule)

    def _w_link_rule(m, k, t):
        if t == 1:
            return m.w[k,1] == m.z[k,1]
        return m.w[k,t] == m.z[k,t] - m.z[k,t-1]
    m.WLink = Constraint(m.K, m.T, rule=_w_link_rule)

    def _unique_completion_rule(m, k):
        return sum(m.w[k,t] for t in m.T) == m.mdone[k]
    m.UniqueCompletion = Constraint(m.K, rule=_unique_completion_rule)

    # Retard (tardiness) via z cumulatif : ∑_{t>d_k} (1 - z_{k,t})
    def tardiness_expr(m, k):
        dk = value(m.d[k])
        return sum(1 - m.z[k,t] for t in m.T if t > dk)

    # Objectif
    if objective == "nb_projects":
        m.OBJ = Objective(expr=sum(m.mdone[k] for k in m.K), sense=maximize)
    elif objective == "on_time":
        m.OBJ = Objective(expr=sum(m.z[k, m.d[k]] for k in m.K), sense=maximize)
    elif objective == "combo":
        expr_profit  = sum(m.g[k] * m.mdone[k] for k in m.K) \
                     - sum(m.p[k] * tardiness_expr(m, k) for k in m.K)
        expr_nproj   = sum(m.mdone[k] for k in m.K)
        expr_ontime  = sum(m.z[k, m.d[k]] for k in m.K)
        m.OBJ = Objective(expr=w_profit*expr_profit + w_nproj*expr_nproj + w_ontime*expr_ontime,
                          sense=maximize)
    else:  # "profit"
        expr_profit = sum(m.g[k] * m.mdone[k] for k in m.K) \
                    - sum(m.p[k] * tardiness_expr(m, k) for k in m.K)
        m.OBJ = Objective(expr=expr_profit, sense=maximize)

    return m

# ---------- Solve + extraction ----------
def solve_model(model, solver_name="glpk", tee=False, timelimit=None):
    opt = SolverFactory(solver_name)
    if opt is None or not opt.available():
        raise RuntimeError(f"Le solveur '{solver_name}' n'est pas disponible.")
    if timelimit is not None:
        # GLPK utilise 'tmlim' ; d'autres solveurs ont d'autres options
        try:
            opt.options["tmlim"] = int(timelimit)
        except Exception:
            pass
    results = opt.solve(model, tee=tee)
    return results

def extract_solution(m):
    sol = {
        "objective": float(value(m.OBJ)),
        "projects_done": {k: int(value(m.mdone[k])) for k in m.K},
        "completion_day": {},
        "on_time": {},
        "assignments": []
    }
    for k in m.K:
        day = None
        for t in m.T:
            if value(m.w[k,t]) > 0.5:
                day = int(t); break
        sol["completion_day"][k] = day
        sol["on_time"][k] = int(value(m.z[k, m.d[k]]))

    for i in m.I:
        for j in m.J:
            for k in m.K:
                for t in m.T:
                    if value(m.x[i,j,k,t]) > 0.5:
                        sol["assignments"].append((i,j,k,int(t)))
    return sol

# ---------- Exécution directe ----------
if __name__ == "__main__":
    H, I, J, K, eta, v, mu, gain, due, pen = load_instance(INSTANCE_PATH)
    m = build_model(H, I, J, K, eta, v, mu, gain, due, pen,
                    objective=OBJECTIVE,
                    w_profit=W_PROFIT, w_nproj=W_NPROJ, w_ontime=W_ONTIME)
    solve_model(m, solver_name=SOLVER_NAME, tee=TEE, timelimit=TIMELIMIT)
    sol = extract_solution(m)

    print("\n=== Résultat ===")
    print(f"Objectif = {sol['objective']:.3f}")
    print("Projets réalisés :", sol["projects_done"])
    print("Jour de complétion :", sol["completion_day"])
    print("À l'heure :", sol["on_time"])
    print(f"Affectations (i,j,k,t) — {len(sol['assignments'])} entrées")
    # Pour lister intégralement :
    # for tup in sol["assignments"]:
    #     print(tup)
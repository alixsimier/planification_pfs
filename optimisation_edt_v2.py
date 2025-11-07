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

def build_model(H, S, Q, P, eta, v, mu, gain, due, pen,
                objective="profit", w_profit=1.0, w_nproj=0.0, w_ontime=0.0):
    m = ConcreteModel("StaffingProjects")

    # Ensembles
    m.S = Set(initialize=S, ordered=True)
    m.Q = Set(initialize=Q, ordered=True)
    m.P = Set(initialize=P, ordered=True)
    m.T = RangeSet(1, H)

    # Paramètres
    m.eta = Param(m.S, m.Q, initialize=lambda m,s,q: int(eta[s][q]) if q in eta[s] else 0, within=Binary, default=0)
    m.v   = Param(m.S, m.T, initialize=lambda m,s,t: int(v[s][t]), within=Binary, default=1)
    m.mu  = Param(m.P, m.Q, initialize=lambda m,p,q: int(mu[p][q]) if q in mu[p] else 0,
                  within=NonNegativeIntegers, default=0)
    m.g   = Param(m.P, initialize=lambda m,p: int(gain[p]), within=NonNegativeIntegers, default=0)
    m.d   = Param(m.P, initialize=lambda m,p: max(1, min(H, int(due[p]))), within=NonNegativeIntegers)
    m.p   = Param(m.P, initialize=lambda m,p: int(pen[p]), within=NonNegativeIntegers, default=0)

    # Variables
    m.x     = Var(m.S, m.Q, m.P, m.T, within=Binary)  # affectations
    m.z     = Var(m.P, m.T, within=Binary)            # complétion cumulée
    m.w     = Var(m.P, m.T, within=Binary)            # complétion exacte au jour t
    m.mdone = Var(m.P, within=Binary)                 # projet réalisé

    # Contraintes
    def _qual_rule(m, s, q, p, t):
        return m.x[s,q,p,t] <= m.eta[s,q]
    m.Qualif = Constraint(m.S, m.Q, m.P, m.T, rule=_qual_rule)

    def _one_task_per_day_rule(m, s, t):
        return sum(m.x[s,q,p,t] for q in m.Q for p in m.P) <= m.v[s,t]
    m.OneTaskPerDay = Constraint(m.S, m.T, rule=_one_task_per_day_rule)

    def _total_cover_rule(m, p, q):
        req = m.mu[p,q]
        if value(req) == 0:
            return Constraint.Skip
        return sum(m.x[s,q,p,t] for s in m.S for t in m.T) >= req * m.mdone[p]
    m.TotalCover = Constraint(m.P, m.Q, rule=_total_cover_rule)

    def _cum_cover_rule(m, p, q, t):
        req = m.mu[p,q]
        if value(req) == 0:
            return Constraint.Skip
        return sum(m.x[s,q,p,tt] for s in m.S for tt in m.T if tt <= t) >= req * m.z[k,t]
    m.CumCover = Constraint(m.P, m.Q, m.T, rule=_cum_cover_rule)

    def _z_mono_rule(m, p, t):
        if t == 1:
            return Constraint.Skip
        return m.z[p,t-1] <= m.z[p,t]
    m.ZMonotone = Constraint(m.P, m.T, rule=_z_mono_rule)

    def _w_link_rule(m, p, t):
        if t == 1:
            return m.w[p,1] == m.z[p,1]
        return m.w[p,t] == m.z[p,t] - m.z[p,t-1]
    m.WLink = Constraint(m.P, m.T, rule=_w_link_rule)

    def _unique_completion_rule(m, p):
        return sum(m.w[p,t] for t in m.T) == m.mdone[k]
    m.UniqueCompletion = Constraint(m.P, rule=_unique_completion_rule)

    # Retard (tardiness) via z cumulatif : ∑_{t>d_p} (1 - z_{p,t})
    def tardiness_expr(m, p):
        dp = value(m.d[p])
        return sum(1 - m.z[p,t] for t in m.T if t > dp)

    # Objectif
    if objective == "nb_projects":
        m.OBJ = Objective(expr=sum(m.mdone[p] for p in m.P), sense=maximize)
    elif objective == "on_time":
        m.OBJ = Objective(expr=sum(m.z[p, m.d[p]] for p in m.P), sense=maximize)
    elif objective == "combo":
        expr_profit  = sum(m.g[p] * m.mdone[p] for p in m.P) \
                     - sum(m.p[p] * tardiness_expr(m, p) for p in m.P)
        expr_nproj   = sum(m.mdone[p] for p in m.P)
        expr_ontime  = sum(m.z[p, m.d[p]] for p in m.P)
        m.OBJ = Objective(expr=w_profit*expr_profit + w_nproj*expr_nproj + w_ontime*expr_ontime,
                          sense=maximize)
    else:  # "profit"
        expr_profit = sum(m.g[p] * m.mdone[p] for p in m.P) \
                    - sum(m.p[p] * tardiness_expr(m, p) for p in m.P)
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
        "projects_done": {p: int(value(m.mdone[k])) for p in m.P},
        "completion_day": {},
        "on_time": {},
        "assignments": []
    }
    for p in m.P:
        day = None
        for t in m.T:
            if value(m.w[p,t]) > 0.5:
                day = int(t); break
        sol["completion_day"][p] = day
        sol["on_time"][p] = int(value(m.z[p, m.d[p]]))

    for s in m.S:
        for q in m.Q:
            for p in m.P:
                for t in m.T:
                    if value(m.x[s,q,p,t]) > 0.5:
                        sol["assignments"].append((s,q,p,int(t)))
    return sol

# ---------- Exécution directe ----------
if __name__ == "__main__":
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(INSTANCE_PATH)
    m = build_model(H, S, Q, P, eta, v, mu, gain, due, pen,
                    objective=OBJECTIVE,
                    w_profit=W_PROFIT, w_nproj=W_NPROJ, w_ontime=W_ONTIME)
    solve_model(m, solver_name=SOLVER_NAME, tee=TEE, timelimit=TIMELIMIT)
    sol = extract_solution(m)

    print("\n=== Résultat ===")
    print(f"Objectif = {sol['objective']:.3f}")
    print("Projets réalisés :", sol["projects_done"])
    print("Jour de complétion :", sol["completion_day"])
    print("À l'heure :", sol["on_time"])
    print(f"Affectations (sp,q,p,t) — {len(sol['assignments'])} entrées")
    # Pour lister intégralement :
    # for tup in sol["assignments"]:
    #     print(tup)
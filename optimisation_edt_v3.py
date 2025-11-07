# Paramètres
INSTANCE_PATH = "instances/medium_instance.json"
OBJECTIVE     = "gain"                          # "gain" | "nb_projects" | "on_time"
SOLVER_NAME   = "glpk"                          # ex: "glpk", "cbc", "gurobi", ...
TEE           = True
TIMELIMIT     = None

# Imports 
import json
from collections import defaultdict
from pyomo.environ import (
    ConcreteModel, Set, RangeSet, Param, Var, Binary, NonNegativeIntegers,
    Objective, Constraint, maximize, value, SolverFactory
)
from load_instance import load_instance

# Modèle 
def build_model(H, S, Q, P, eta, v, mu, gain, due, pen, objective="gain"):
    m = ConcreteModel("PFS_Planification_SQP")

    m.S = Set(initialize=S, ordered=True)     # staff
    m.Q = Set(initialize=Q, ordered=True)     # qualifications
    m.P = Set(initialize=P, ordered=True)     # projets
    m.T = RangeSet(1, H)                      # horizon {1..H}

    m.eta = Param(m.S, m.Q, initialize=lambda m,s,q: int(eta[s][q]) if q in eta[s] else 0,
                  within=Binary, default=0)                                                 # η_{s,q}
    m.v   = Param(m.S, m.T, initialize=lambda m,s,t: int(v[s][t]), within=Binary, default=0)  # v_{s,t}
    m.mu  = Param(m.P, m.Q, initialize=lambda m,p,q: int(mu[p][q]) if q in mu[p] else 0,
                  within=NonNegativeIntegers, default=0)                                     # μ_{p,q}
    m.g   = Param(m.P, initialize=lambda m,p: int(gain[p]), within=NonNegativeIntegers, default=0)  # g_p
    m.d   = Param(m.P, initialize=lambda m,p: max(1, min(H, int(due[p]))), within=NonNegativeIntegers) # d_p
    m.p   = Param(m.P, initialize=lambda m,p_: int(pen[p_]), within=NonNegativeIntegers, default=100)   # p_p

    # Variables (mêmes que le PDF)
    m.lmbda = Var(m.S, m.Q, m.P, m.T, within=Binary, initialize=0)          # λ_{s,q,p,t}
    m.z     = Var(m.P, m.T, within=Binary, initialize=0)                    # z_{p,t}
    m.m     = Var(m.P, within=Binary, initialize=0)                         # m_p
    m.tau   = Var(m.P, within=NonNegativeIntegers, initialize=0)            # τ_p (déclarée ; non utilisée si non contrainte)

    # --- Contraintes du PDF ---
    # (i) Qualification : λ_{s,q,p,t} ≤ η_{s,q}
    def _qual_rule(m, s, q, p, t):
        return m.lmbda[s,q,p,t] <= m.eta[s,q]
    m.Qualif = Constraint(m.S, m.Q, m.P, m.T, rule=_qual_rule)

    # (ii) Unicité / vacances : ∑_{q,p} λ_{s,q,p,t} ≤ v_{s,t}
    def _one_task_day_rule(m, s, t):
        return sum(m.lmbda[s,q,p,t] for q in m.Q for p in m.P) <= m.v[s,t]
    m.OneTaskPerDay = Constraint(m.S, m.T, rule=_one_task_day_rule)

    # (iii) Réalisation globale (forme ≥ m_p - 1) : ∑_{s,t} λ_{s,q,p,t} - μ_{p,q} ≥ m_p - 1
    def _cover_total_rule(m, p, q):
        return (sum(m.lmbda[s,q,p,t] for s in m.S for t in m.T) - m.mu[p,q]) >= (m.m[p] - 1)
    m.CoverTotal = Constraint(m.P, m.Q, rule=_cover_total_rule)

    # (iv) Réalisation cumulée (forme ≥ z_{p,t} - 1) : ∑_{s,τ≤t} λ_{s,q,p,τ} - μ_{p,q} ≥ z_{p,t} - 1
    def _cover_cum_rule(m, p, q, t):
        return (sum(m.lmbda[s,q,p,tt] for s in m.S for tt in m.T if tt <= t) - m.mu[p,q]) >= (m.z[p,t] - 1)
    m.CoverCum = Constraint(m.P, m.Q, m.T, rule=_cover_cum_rule)

    # --- Objectifs ---
    if objective == "nb_projects":
        m.OBJ = Objective(expr=sum(m.m[p] for p in m.P), sense=maximize)
    elif objective == "on_time":
        m.OBJ = Objective(expr=sum(m.z[p, m.d[p]] for p in m.P), sense=maximize)
    else:  # "gain" par défaut
        m.OBJ = Objective(
            expr=sum(m.g[p]*m.m[p] for p in m.P)
               - sum(m.p[p] * sum(1 - m.z[p,t] for t in m.T if t > m.d[p]) for p in m.P),
            sense=maximize
        )

    return m

# ---------- Solve & extraction ----------
def solve_model(model, solver_name="glpk", tee=False, timelimit=None):
    opt = SolverFactory(solver_name)
    if opt is None or not opt.available():
        raise RuntimeError(f"Le solveur '{solver_name}' n'est pas disponible.")
    if timelimit is not None:
        try:
            opt.options["tmlim"] = int(timelimit)  # GLPK
        except Exception:
            pass
    results = opt.solve(model, tee=tee)
    return results

def extract_solution(m):
    sol = {
        "objective": float(value(m.OBJ)),
        "projects_done": {p: int(value(m.m[p])) for p in m.P},
        "on_time": {p: int(value(m.z[p, m.d[p]])) for p in m.P},
        "completion_day": {},   # premier t tel que z_{p,t}=1 (si existe)
        "assignments": []       # tuples (s,q,p,t) où λ=1
    }
    for p in m.P:
        day = None
        for t in m.T:
            if value(m.z[p,t]) > 0.5:
                day = int(t); break
        sol["completion_day"][p] = day

    for s in m.S:
        for q in m.Q:
            for p in m.P:
                for t in m.T:
                    if value(m.lmbda[s,q,p,t]) > 0.5:
                        sol["assignments"].append((s,q,p,int(t)))
    return sol

# ---------- Exécution ----------
if __name__ == "__main__":
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance(INSTANCE_PATH)
    m = build_model(H, S, Q, P, eta, v, mu, gain, due, pen, objective=OBJECTIVE)
    solve_model(m, solver_name=SOLVER_NAME, tee=TEE, timelimit=TIMELIMIT)
    sol = extract_solution(m)

    print("\n=== Résultat (notations S/Q/P conformes à load_instance) ===")
    print(f"Objectif = {sol['objective']:.3f}")
    print("Projets réalisés (m_p) :", sol["projects_done"])
    print("À l'heure (z_{p,d_p})  :", sol["on_time"])
    print("Jour de complétion (1er t avec z=1) :", sol["completion_day"])
    print(f"Affectations λ (s,q,p,t) — {len(sol['assignments'])} entrées")

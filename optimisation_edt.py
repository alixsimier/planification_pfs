# planif_pyomo.py
# -*- coding: utf-8 -*-
"""
Modèle Pyomo pour la planification personnel/projets avec compétences et échéances.

Référence de modélisation (sets/vars/constraints/objets) : pfs_planification.pdf.  # voir conversation
Contexte/format d'instances JSON (toy/medium/large) : Projet_fin_sequence_ORRA_2025_26-1.pdf.  # voir conversation

JSON attendu (exemple minimal) :
{
  "H": 15,                                  # horizon (nb de jours ouvrés indexés 1..H)
  "people": ["alice","bob"],
  "skills": ["A","B","C"],
  "projects": ["I","II"],
  "eta": { "alice": {"A":1,"B":0,"C":1}, "bob":{"A":1,"B":1,"C":0} },   # η_{i,j} ∈ {0,1}
  "vacations": { "alice": [3,4], "bob":[10] },                           # jours non travaillés -> v_{i,t}=0
  "mu": { "I":{"A":6,"B":2,"C":5}, "II":{"A":3,"B":0,"C":2} },           # μ_{k,j} (0 si compétence non requise)
  "gain": { "I": 120, "II": 60 },                                        # g_k
  "due":  { "I": 10,  "II": 7 },                                         # d_k (jour dans 1..H)
  "penalty": { "I": 8, "II": 5 }                                         # p_k (par jour de retard)
}

Objectifs disponibles (--objective) :
  - "profit" (défaut) : max sum_k g_k m_k - sum_k p_k * sum_{t>d_k} (1 - z_{k,t})
  - "nb_projects"     : max sum_k m_k
  - "on_time"         : max sum_k z_{k,d_k}      (livrés à l'heure)
  - "combo"           : combinaison pondérée via --w_profit --w_nproj --w_ontime
"""

import json
import argparse
from collections import defaultdict

from pyomo.environ import (
    ConcreteModel, Set, RangeSet, Param, Var, Binary, NonNegativeIntegers, NonNegativeReals,
    Objective, Constraint, maximize, value, summation, SolverFactory
)

def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    H = int(data["H"])
    I = list(data["people"])
    J = list(data["skills"])
    K = list(data["projects"])

    # η_{i,j}
    eta = defaultdict(lambda: defaultdict(int))
    for i, row in data.get("eta", {}).items():
        for j, v in row.items():
            eta[i][j] = int(v)

    # v_{i,t} : 1 si disponible, 0 si congé
    vacations = {i: set(map(int, days)) for i, days in data.get("vacations", {}).items()}
    v = defaultdict(lambda: defaultdict(int))
    for i in I:
        for t in range(1, H+1):
            v[i][t] = 0 if t in vacations.get(i, set()) else 1

    # μ_{k,j}
    mu = defaultdict(lambda: defaultdict(int))
    for k, row in data.get("mu", {}).items():
        for j, req in row.items():
            mu[k][j] = int(req)

    # gains, due dates, penalités
    gain = {k: int(data.get("gain", {}).get(k, 0)) for k in K}
    due  = {k: int(data.get("due",  {}).get(k, H)) for k in K}
    pen  = {k: int(data.get("penalty", {}).get(k, 0)) for k in K}

    return H, I, J, K, eta, v, mu, gain, due, pen


def build_model(H, I, J, K, eta, v, mu, gain, due, pen,
                objective="profit",
                w_profit=1.0, w_nproj=0.0, w_ontime=0.0):
    m = ConcreteModel("StaffingProjects")

    # Sets
    m.I = Set(initialize=I, ordered=True)
    m.J = Set(initialize=J, ordered=True)
    m.K = Set(initialize=K, ordered=True)
    m.T = RangeSet(1, H)

    # Params
    m.eta = Param(m.I, m.J, initialize=lambda m,i,j: int(eta[i][j]) if j in eta[i] else 0, within=Binary, default=0)
    m.v   = Param(m.I, m.T, initialize=lambda m,i,t: int(v[i][t]), within=Binary, default=1)
    m.mu  = Param(m.K, m.J, initialize=lambda m,k,j: int(mu[k][j]) if j in mu[k] else 0,
                  within=NonNegativeIntegers, default=0)
    m.g   = Param(m.K, initialize=lambda m,k: int(gain[k]), within=NonNegativeIntegers, default=0)
    # borne la due-date dans [1, H] par prudence
    m.d   = Param(m.K, initialize=lambda m,k: max(1, min(H, int(due[k]))), within=NonNegativeIntegers)
    m.p   = Param(m.K, initialize=lambda m,k: int(pen[k]), within=NonNegativeIntegers, default=0)

    # Variables
    # x_{i,j,k,t} : i affecté à la compétence j du projet k au jour t
    m.x = Var(m.I, m.J, m.K, m.T, within=Binary)

    # z_{k,t} : projet k complété (cumulatif) au plus tard au jour t
    m.z = Var(m.K, m.T, within=Binary)

    # w_{k,t} : “événement” de complétion exactement au jour t (saut de z)
    m.w = Var(m.K, m.T, within=Binary)

    # m_k : projet réalisé (une fois au plus)
    m.mdone = Var(m.K, within=Binary)

    # ---------- Contraintes ----------

    # (1) Qualification : x <= eta
    def _qual_rule(m, i, j, k, t):
        return m.x[i,j,k,t] <= m.eta[i,j]
    m.Qualif = Constraint(m.I, m.J, m.K, m.T, rule=_qual_rule)

    # (2) Unicité / dispo par personne et par jour : sum_{j,k} x_{i,j,k,t} <= v_{i,t}
    def _one_task_per_day_rule(m, i, t):
        return sum(m.x[i,j,k,t] for j in m.J for k in m.K) <= m.v[i,t]
    m.OneTaskPerDay = Constraint(m.I, m.T, rule=_one_task_per_day_rule)

    # (3) Couverture totale par compétence si projet réalisé : sum_{i,t} x_{i,j,k,t} >= mu_{k,j} * m_k
    def _total_cover_rule(m, k, j):
        req = m.mu[k,j]
        if value(req) == 0:
            # pas de contrainte si la comp. n'est pas requise
            return Constraint.Skip
        return sum(m.x[i,j,k,t] for i in m.I for t in m.T) >= req * m.mdone[k]
    m.TotalCover = Constraint(m.K, m.J, rule=_total_cover_rule)

    # (4) Couverture cumulée à la date t si z_{k,t}=1 :
    #     sum_{i,τ<=t} x_{i,j,k,τ} >= mu_{k,j} * z_{k,t}  (pour chaque comp. requise)
    def _cum_cover_rule(m, k, j, t):
        req = m.mu[k,j]
        if value(req) == 0:
            return Constraint.Skip
        return sum(m.x[i,j,k,tt] for i in m.I for tt in m.T if tt <= t) >= req * m.z[k,t]
    m.CumCover = Constraint(m.K, m.J, m.T, rule=_cum_cover_rule)

    # (5) Monotonicité de z : z_{k,t-1} <= z_{k,t}
    def _z_mono_rule(m, k, t):
        if t == 1:
            return Constraint.Skip
        return m.z[k,t-1] <= m.z[k,t]
    m.ZMonotone = Constraint(m.K, m.T, rule=_z_mono_rule)

    # (6) Lien w/z :
    # w_{k,1} = z_{k,1}
    # w_{k,t} = z_{k,t} - z_{k,t-1}  pour t>=2
    def _w_link_rule(m, k, t):
        if t == 1:
            return m.w[k,1] == m.z[k,1]
        return m.w[k,t] == m.z[k,t] - m.z[k,t-1]
    m.WLink = Constraint(m.K, m.T, rule=_w_link_rule)

    # (7) Unicité de réalisation : sum_t w_{k,t} = m_k  (au plus une fois)
    def _unique_completion_rule(m, k):
        return sum(m.w[k,t] for t in m.T) == m.mdone[k]
    m.UniqueCompletion = Constraint(m.K, rule=_unique_completion_rule)

    # ---------- Fonctions d’aide pour le retard ----------
    # Tardiness “linéaire” via z cumulatif : sum_{t=d_k+1..H} (1 - z_{k,t})
    def tardiness_expr(m, k):
        dk = value(m.d[k])
        return sum(1 - m.z[k,t] for t in m.T if t > dk)

    # ---------- Objectifs ----------
    if objective == "nb_projects":
        m.OBJ = Objective(expr=sum(m.mdone[k] for k in m.K), sense=maximize)

    elif objective == "on_time":
        # max nombre de projets livrés à la due date
        # (si due date > H : on borne à H par notre Param m.d)
        m.OBJ = Objective(expr=sum(m.z[k, m.d[k]] for k in m.K), sense=maximize)

    elif objective == "combo":
        expr_profit  = sum(m.g[k] * m.mdone[k] for k in m.K) \
                       - sum(m.p[k] * tardiness_expr(m, k) for k in m.K)
        expr_nproj   = sum(m.mdone[k] for k in m.K)
        expr_ontime  = sum(m.z[k, m.d[k]] for k in m.K)
        m.OBJ = Objective(expr=w_profit*expr_profit + w_nproj*expr_nproj + w_ontime*expr_ontime,
                          sense=maximize)

    else:  # "profit" par défaut
        expr_profit = sum(m.g[k] * m.mdone[k] for k in m.K) \
                      - sum(m.p[k] * tardiness_expr(m, k) for k in m.K)
        m.OBJ = Objective(expr=expr_profit, sense=maximize)

    return m


def solve_model(model, solver_name="glpk", tee=False, timelimit=None):
    opt = SolverFactory(solver_name)
    if opt is None or not opt.available():
        raise RuntimeError(f"Le solveur '{solver_name}' n'est pas disponible. Installe-le ou choisis-en un autre.")
    if timelimit is not None:
        # conventions usuelles (GLPK: 'tmlim', CBC/Gurobi/CPLEX ont d'autres noms)
        try:
            opt.options["tmlim"] = int(timelimit)
        except Exception:
            pass
    results = opt.solve(model, tee=tee)
    return results


def extract_solution(m):
    # renvoie un dict propre à logguer ou sérialiser
    sol = {
        "objective": float(value(m.OBJ)),
        "projects_done": {k: int(value(m.mdone[k])) for k in m.K},
        "completion_day": {},     # jour de complétion détecté via w
        "on_time": {},            # bool: z[k, d_k]
        "assignments": []         # (i,j,k,t) avec x=1
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=str, help="chemin vers l'instance JSON")
    parser.add_argument("--objective", type=str, default="profit",
                        choices=["profit","nb_projects","on_time","combo"])
    parser.add_argument("--w_profit", type=float, default=1.0)
    parser.add_argument("--w_nproj",  type=float, default=0.0)
    parser.add_argument("--w_ontime", type=float, default=0.0)
    parser.add_argument("--solver", type=str, default="glpk")
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--timelimit", type=int, default=None)
    args = parser.parse_args()

    H, I, J, K, eta, v, mu, gain, due, pen = load_instance(args.instance)
    m = build_model(H, I, J, K, eta, v, mu, gain, due, pen,
                    objective=args.objective,
                    w_profit=args.w_profit, w_nproj=args.w_nproj, w_ontime=args.w_ontime)
    solve_model(m, solver_name=args.solver, tee=args.tee, timelimit=args.timelimit)
    sol = extract_solution(m)

    print("\n=== Résultat ===")
    print(f"Objectif = {sol['objective']:.3f}")
    print("Projets réalisés :", sol["projects_done"])
    print("Jour de complétion :", sol["completion_day"])
    print("À l'heure :", sol["on_time"])
    print(f"Affectations (i,j,k,t) — {len(sol['assignments'])} entrées")
    # décommente pour lister intégralement :
    # for tup in sol["assignments"]:
    #     print(tup)

if __name__ == "__main__":
    main()
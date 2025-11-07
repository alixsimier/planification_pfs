from collections import defaultdict
import json

def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format avec 'horizon', 'qualifications', 'staff', 'jobs'
    H = int(data["horizon"])
    Q = list(data["qualifications"])

    # Personnel / qualifications / vacations
    S = []
    eta = defaultdict(lambda: defaultdict(int))   # η_{i,j}
    v   = defaultdict(lambda: defaultdict(int))   # v_{i,t} (1 si dispo)
    vacations_map = {}

    for s in data.get("staff", []):
        name = s["name"]
        S.append(name)
        quals = set(s.get("qualifications", []))
        for q in Q:
            eta[name][q] = 1 if q in quals else 0
        vacations_map[name] = set(int(d) for d in s.get("vacations", []))

    for s in S:
        for t in range(1, H + 1):
            v[s][t] = 0 if t in vacations_map.get(s, set()) else 1

    # Jobs -> projets
    P = []
    mu   = defaultdict(lambda: defaultdict(int))  # μ_{k,j}
    gain = {}
    due  = {}
    pen  = {}

    for job in data.get("jobs", []):
        p = job["name"]
        P.append(p)
        gain[p] = int(job.get("gain", 0))
        due[p]  = int(job.get("due_date", H))
        pen[p]  = int(job.get("daily_penalty", 0))
        for q, req in job.get("working_days_per_qualification", {}).items():
            mu[p][q] = int(req)

    return H, S, Q, P, eta, v, mu, gain, due, pen

if __name__ == "__main__":
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance("instances/toy_instance.json")
    print(H, S, Q, P, eta, v, mu, gain, due, pen)
from collections import defaultdict
import json

def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format avec 'horizon', 'qualifications', 'staff', 'jobs'
    H = int(data["horizon"])
    J = list(data["qualifications"])

    # Personnel / qualifications / vacations
    I = []
    eta = defaultdict(lambda: defaultdict(int))   # η_{i,j}
    v   = defaultdict(lambda: defaultdict(int))   # v_{i,t} (1 si dispo)
    vacations_map = {}

    for s in data.get("staff", []):
        name = s["name"]
        I.append(name)
        quals = set(s.get("qualifications", []))
        for j in J:
            eta[name][j] = 1 if j in quals else 0
        vacations_map[name] = set(int(d) for d in s.get("vacations", []))

    for i in I:
        for t in range(1, H + 1):
            v[i][t] = 0 if t in vacations_map.get(i, set()) else 1

    # Jobs -> projets
    K = []
    mu   = defaultdict(lambda: defaultdict(int))  # μ_{k,j}
    gain = {}
    due  = {}
    pen  = {}

    for job in data.get("jobs", []):
        k = job["name"]
        K.append(k)
        gain[k] = int(job.get("gain", 0))
        due[k]  = int(job.get("due_date", H))
        pen[k]  = int(job.get("daily_penalty", 0))
        for j, req in job.get("working_days_per_qualification", {}).items():
            mu[k][j] = int(req)

    return H, I, J, K, eta, v, mu, gain, due, pen

if __name__ == "__main__":
    H, I, J, K, eta, v, mu, gain, due, pen = load_instance("instances/toy_instance.json")
    print(H, I, J, K, eta, v, mu, gain, due, pen)
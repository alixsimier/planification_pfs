import json

def load_instance(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Horizon et qualifications
    H = int(data["horizon"])
    Q = list(data["qualifications"])

    # Staff
    S = []
    eta = {}
    v = {}
    vacations_map = {}

    for s in data.get("staff", []):
        name = s["name"]
        S.append(name)
        quals = set(s.get("qualifications", []))
        eta[name] = {q: 1 if q in quals else 0 for q in Q}

        # Vacations mapping
        vacations_map[name] = set(s.get("vacations", []))

        # Disponibilité par jour
        v[name] = {}
        for t in range(1, H + 1):
            v[name][t] = 0 if t in vacations_map[name] else 1

    # Jobs / Projets
    P = []
    mu = {}
    gain = {}
    due = {}
    pen = {}

    for job in data.get("jobs", []):
        p = job["name"]
        P.append(p)
        gain[p] = int(job.get("gain", 0))
        due[p] = int(job.get("due_date", H))
        pen[p] = int(job.get("daily_penalty", 0))

        # Besoins par qualification
        req = job.get("working_days_per_qualification", {})
        mu[p] = {q: int(req[q]) if q in req else 0 for q in Q}

    return H, S, Q, P, eta, v, mu, gain, due, pen

# Exemple d'utilisation
if __name__ == "__main__":
    H, S, Q, P, eta, v, mu, gain, due, pen = load_instance("instances/toy_instance.json")
    print("Horizon:", H)
    print("Staff:", S)
    print("Qualifications:", Q)
    print("Jobs:", P)
    print("Eta:", eta)
    print("Disponibilité:", v)
    print("Mu:", mu)
    print("Gains:", gain)
    print("Due dates:", due)
    print("Penalties:", pen)

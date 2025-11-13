import json
import random
import string
from rich import print as rprint
from faker import Faker
import json
import os

fake = Faker()

def load_instance(path="instances/medium_instance.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = data
    return new_data

def new_horizon(data, changement=2):
    c = random.randint(-changement, changement)
    data["horizon"] = int(data["horizon"]) + c
    return data

def new_qualifications(data, changement=2):
    qualifications = data["qualifications"].copy()
    c = random.randint(-changement, changement)
    if c > 0:
        available = [q for q in string.ascii_uppercase if q not in qualifications]
        new_quals = random.sample(available, min(c, len(available)))
        qualifications.extend(new_quals)

    elif c < 0:
        remove_count = min(abs(c), len(qualifications) - 1)
        to_remove = random.sample(qualifications, remove_count)
        qualifications = [q for q in qualifications if q not in to_remove]
    data["qualifications"] = qualifications
    return data

def new_staff(data, changement=2):
    staff = data["staff"].copy()
    c = random.randint(-changement, changement)
    if c < 0:
        remove_count = min(abs(c), len(staff) - 1)
        to_remove = random.sample(staff, remove_count)
        staff = [q for q in staff if q not in to_remove]
    elif c>0:
        for i in range(c):
            new = {
                "name": fake.first_name(),
                "qualifications": random.sample(data["qualifications"], random.randint(1, len(data["qualifications"]))),
                "vacations": random.sample(range(1, data["horizon"]+1), random.randint(0, min(5, data["horizon"])))
            }
            staff.append(new)
    data["staff"]=staff
    return data

def new_job(data, changement=2):
    jobs = data["jobs"].copy()
    c = random.randint(-changement, changement)
    if c < 0:
        remove_count = min(abs(c), len(jobs) - 1)
        to_remove = random.sample(jobs, remove_count)
        jobs = [q for q in jobs if q not in to_remove]
    elif c>0:
        for i in range(c):
            new = {
                "name": f"Job{len(jobs)+i+1}",
                "gain": random.randint(min([j['gain'] for j in jobs]), max([j['gain'] for j in jobs])),
                "due_date": random.randint(min([j['due_date'] for j in jobs]), max([j['due_date'] for j in jobs])),
            }
            remaining_days = new["due_date"]
            num_qual = random.randint(min([len(j['working_days_per_qualification']) for j in jobs]), max([len(j['working_days_per_qualification']) for j in jobs]))
            job_quals = random.sample(data["qualifications"], num_qual)
            remaining_days = new["due_date"]
            working_days_per_qualification = {}
            for idx, q in enumerate(job_quals):
                remaining_qual = len(job_quals) - idx
                max_days = max(1, remaining_days - (remaining_qual - 1))
                days = random.randint(max(1, max_days//2), max_days)
                working_days_per_qualification[q] = days
                remaining_days -= days
            new["working_days_per_qualification"] = working_days_per_qualification
            jobs.append(new)
    data["jobs"]=jobs
    return data

def generate_instance(input_path="instances/medium_instance.json", 
                      output_path="instances/generated_instance.json", 
                      changement=2):
    data = load_instance(input_path)
    data = new_horizon(data, changement)
    data = new_qualifications(data, changement)
    data = new_staff(data, changement)
    data = new_job(data, changement)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Nouvelle instance générée et sauvegardée dans {output_path}")
    return data

def generate_multiple_instances(n=100, 
                                input_path="instances/medium_instance.json", 
                                output_dir="instances/generated_instances", 
                                changement=2):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(1, n+1):
        output_path = os.path.join(output_dir, f"instance_{i}.json")
        generate_instance(input_path=input_path, 
                          output_path=output_path, 
                          changement=changement)
    print(f"{n} instances générées dans le dossier {output_dir}")

if __name__ == "__main__":
    generate_multiple_instances(input_path="instances/generation_instance.json")

    # data = load_instance()
    # data = new_horizon(data)
    # data = new_qualifications(data)
    # data = new_staff(data)
    # data = new_job(data)
    # rprint(data)

"""Build a summary JSON with teacher/student ROUGE scores and retention."""
import json, glob, os

LANGUAGES = ["amharic", "hausa", "nepali", "pashto"]

def get_test_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_latest_dir(base, pattern):
    matches = glob.glob(os.path.join(base, pattern))
    return max(matches, key=os.path.getmtime) if matches else None

results = {}

for lang in LANGUAGES:
    teacher_dir = find_latest_dir("./teachers", f"{lang}_teacher_*")
    student_dir = find_latest_dir("./students", f"{lang}_student_fast_*")

    t = get_test_results(os.path.join(teacher_dir, "test_results.json"))
    s = get_test_results(os.path.join(student_dir, "test_results.json"))

    tr1, tr2, trL = t["test_rouge1"], t["test_rouge2"], t["test_rougeL"]
    sr1, sr2, srL = s["test_rouge1"], s["test_rouge2"], s["test_rougeL"]

    results[lang] = {
        "teacher": {
            "rouge1": round(tr1, 6),
            "rouge2": round(tr2, 6),
            "rougeL": round(trL, 6),
        },
        "student": {
            "rouge1": round(sr1, 6),
            "rouge2": round(sr2, 6),
            "rougeL": round(srL, 6),
        },
        "retention": {
            "rouge1": round(sr1 / tr1 * 100, 2),
            "rouge2": round(sr2 / tr2 * 100, 2),
            "rougeL": round(srL / trL * 100, 2),
        }
    }

output_path = "./training_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print(f"\nSaved to {output_path}")

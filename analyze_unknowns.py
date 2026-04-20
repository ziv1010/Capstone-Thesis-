import json
import glob
from collections import Counter

def normalise_outcome(label):
    if not label:
        return "unknown_null"
    l = str(label).lower()
    if "appellant_won" in l or "petitioner_won" in l or "_won" in l:
        return "win"
    if "appellant_lost" in l or "respondent_won" in l or "_lost" in l:
        return "loss"
    return "unknown_other"

buckets = [
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/family_matrimonial_timed_mistral",
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/fin_fraud_timed_mistral",
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/land_property_timed_mistral",
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/motor_accidents_timed_mistral",
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/sexual_offences_timed_mistral"
]

unknown_labels = Counter()
null_count = 0
total_cases = 0

for b_idx, b in enumerate(buckets):
    files = glob.glob(b + "/*.json")
    for idx, f in enumerate(files):
        if f.endswith("report.json"): continue
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                label = data.get("case_outcome_label")
                total_cases += 1
                norm = normalise_outcome(label)
                if norm == "unknown_null":
                    null_count += 1
                elif norm == "unknown_other":
                    unknown_labels[str(label)] += 1
        except Exception:
            pass

print(f"Total cases read: {total_cases}")
print(f"Cases with purely missing/null labels: {null_count}")
print("\nMost common string labels categorized as 'Unknown':")
for label, count in unknown_labels.most_common(50):
    print(f"{count:5d} : {label}")

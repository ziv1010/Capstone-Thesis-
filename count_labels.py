import json
import glob
from collections import Counter

buckets = [
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/family_matrimonial_timed_mistral",
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/fin_fraud_timed_mistral",
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/land_property_timed_mistral",
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/motor_accidents_timed_mistral",
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/sexual_offences_timed_mistral"
]

counts = Counter()

for b in buckets:
    files = glob.glob(b + "/*.json")
    for f in files[:2000]:  # sample 2k per bucket (10k total)
        if 'report.json' in f: continue
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                v = data.get("case_outcome_label")
                counts[str(v)] += 1
        except Exception:
            pass

print("Sampled label counts:")
for k, v in counts.most_common(50):
    print(f"{v:6d} : {k}")

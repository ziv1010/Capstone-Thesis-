import os
import json
import collections

base_dir = "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Thesis_FINAL_DATA/experiments_results"
results = collections.defaultdict(dict)
models = os.listdir(base_dir)

for model in models:
    model_dir = os.path.join(base_dir, model)
    if os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith(".json"):
                bucket = file.replace(".json", "")
                filepath = os.path.join(model_dir, file)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    
                    if "aggregate" in data and "macro_f1_mean" in data["aggregate"]:
                        results[bucket][model] = data["aggregate"]["macro_f1_mean"]
                except Exception as e:
                    pass

# Print summary table
if results:
    buckets = list(results.keys())
    all_models = set()
    for b in buckets:
        all_models.update(results[b].keys())
    all_models = sorted(list(all_models))
    
    # header
    print(f"{'Bucket':<30} | " + " | ".join([f"{m:<20}" for m in all_models]))
    print("-" * 150)
    
    for b in sorted(buckets):
        row = f"{b:<30} | "
        for m in all_models:
            val = results[b].get(m, "N/A")
            if isinstance(val, float):
                row += f"{val:<20.4f} | "
            else:
                row += f"{val:<20} | "
        print(row)

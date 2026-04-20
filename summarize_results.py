import os
import json

base_dir = "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Thesis_FINAL_DATA/experiments_results"
results = {}

for root, dirs, files in os.walk(base_dir):
    if "kfold_summary.json" in files:
        filepath = os.path.join(root, "kfold_summary.json")
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                
            # extract useful stuff
            # Typically structure is:
            # {
            #    "overall": {"mean_test_f1_macro": 0.55, ...},
            #    "config": {...}
            # }
            # Or data directly has metric dicts. 
            # Let's print the dict keys or specific fields if they exist
            
            # Since we don't know the exact structure, let's just get whatever has 'f1' or 'macro'
            mean_test_f1 = None
            if "overall" in data and "mean_test_f1_macro" in data["overall"]:
                mean_test_f1 = data["overall"]["mean_test_f1_macro"]
            elif "mean_test_f1_macro" in data:
                mean_test_f1 = data["mean_test_f1_macro"]
            else:
                for k, v in data.items():
                    if isinstance(v, dict) and "mean_test_f1_macro" in v:
                        mean_test_f1 = v["mean_test_f1_macro"]
                        break
            
            # The path could be base_dir/baseline/fin_fraud/kfold/kfold_summary.json
            rel_path = os.path.relpath(root, base_dir)
            results[rel_path] = mean_test_f1 if mean_test_f1 is not None else "N/A"
        except Exception as e:
            pass

for k, v in sorted(results.items()):
    print(f"{k}: {v}")

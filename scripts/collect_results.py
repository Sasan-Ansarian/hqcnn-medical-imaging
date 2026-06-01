import json
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("outputs")

rows = []

for summary_file in OUTPUT_DIR.glob("*/summary.json"):
    exp_name = summary_file.parent.name

    with open(summary_file) as f:
        data = json.load(f)

    rows.append({
        "experiment": exp_name,
        "best_val_acc": data.get("best_val_acc"),
        "test_acc": data.get("test_acc"),
        "test_macro_f1": data.get("test_macro_f1"),
        "runtime_sec": data.get("runtime_sec"),
    })

df = pd.DataFrame(rows)
df = df.sort_values("test_macro_f1", ascending=False)

print("\n=== Experiment Results ===\n")
print(df.to_string(index=False))

# Save table
df.to_csv("outputs/results_table.csv", index=False)

print("\nSaved table to outputs/results_table.csv")

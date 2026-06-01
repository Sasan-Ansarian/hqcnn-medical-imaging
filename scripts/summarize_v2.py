import json
import re
from pathlib import Path
import pandas as pd

rows = []
for p in Path("outputs").glob("*/summary.json"):
    exp = p.parent.name
    with open(p, "r") as f:
        s = json.load(f)

    # parse model + seed from experiment name: dermamnist_<model>_v2_seed<seed>
    m = re.match(r"dermamnist_(.+)_v2_seed(\d+)$", exp)
    if not m:
        continue
    model, seed = m.group(1), int(m.group(2))

    rows.append({
        "experiment": exp,
        "model": model,
        "seed": seed,
        "best_val_acc": s.get("best_val_acc"),
        "test_acc": s.get("test_acc"),
        "test_macro_f1": s.get("test_macro_f1"),
        "runtime_sec": s.get("runtime_sec"),
    })

df = pd.DataFrame(rows).sort_values(["model", "seed"])
print("\n=== V2 Runs (per seed) ===\n")
print(df.to_string(index=False))

grp = df.groupby("model").agg(
    n=("seed", "count"),
    test_acc_mean=("test_acc", "mean"),
    test_acc_std=("test_acc", "std"),
    test_macro_f1_mean=("test_macro_f1", "mean"),
    test_macro_f1_std=("test_macro_f1", "std"),
    runtime_mean=("runtime_sec", "mean"),
).reset_index()

print("\n=== V2 Summary (mean ± std) ===\n")
print(grp.to_string(index=False))

grp.to_csv("outputs/v2_summary_mean_std.csv", index=False)
print("\nSaved: outputs/v2_summary_mean_std.csv")

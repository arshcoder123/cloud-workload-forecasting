import argparse
import numpy as np
import pandas as pd
from .data import load, build_targets, split, HORIZON
from .models import BASELINES, STATISTICAL
from .validate import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--output", default="predictions.csv")
    a = ap.parse_args()

    cpu, _ = load(a.data)
    y = build_targets(cpu)
    train, holdout_ts = split(y, HORIZON)
    models = {**BASELINES, **STATISTICAL}

    out = pd.DataFrame({"timestamp": holdout_ts.to_numpy()})
    for col in ["active_vm_count", "aggregate_cpu_utilization"]:
        vals = train[col].to_numpy()
        ranked = compare(vals, models, HORIZON)
        tied = np.allclose(ranked["mae"], ranked["mae"].iloc[0], atol=1e-9)
        best = "persistence" if tied else ranked.index[0]
        print(f"{col}: selected {best}")
        out[col] = np.asarray(models[best](vals, len(holdout_ts)), float)
    out["active_vm_count"] = out["active_vm_count"].clip(lower=0)
    out["aggregate_cpu_utilization"] = out["aggregate_cpu_utilization"].clip(0, 100)
    out.to_csv(a.output, index=False)
    print("wrote", a.output, out.shape)


if __name__ == "__main__":
    main()
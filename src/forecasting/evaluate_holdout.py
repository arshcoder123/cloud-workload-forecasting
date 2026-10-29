import argparse
import pandas as pd
from .data import load, build_targets, split, HORIZON
from .models import persistence
from .validate import mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data")
    ap.add_argument("--predictions", default="predictions.csv")
    a = ap.parse_args()

    cpu, _ = load(a.data)
    y = build_targets(cpu)
    train, holdout_ts = split(y, HORIZON)
    actual = y.loc[holdout_ts]
    pred = pd.read_csv(a.predictions).set_index("timestamp")

    assert list(pred.index) == list(holdout_ts), "timestamps do not match"

    for col in actual.columns:
        a_, f_ = actual[col].to_numpy(), pred[col].to_numpy()
        base = persistence(train[col].to_numpy(), HORIZON)
        row = {"model_mae": mae(a_, f_), "model_rmse": rmse(a_, f_),
               "persist_mae": mae(a_, base), "persist_rmse": rmse(a_, base)}
        if row["persist_mae"] > 1e-9:
            row["improvement_vs_persistence_%"] = 100 * (1 - row["model_mae"] / row["persist_mae"])
        print(f"\n{col}")
        for k, v in row.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
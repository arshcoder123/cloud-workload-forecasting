import argparse
from .data import load, build_targets, split, HORIZON
from .models import BASELINES, STATISTICAL, FALLBACKS
from .validate import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="challenge_data")
    a = ap.parse_args()

    cpu, _ = load(a.data)
    y = build_targets(cpu)
    train, _ = split(y)                      # holdout values never loaded into train
    print("train length:", len(train), "last timestamp:", train.index[-1])  # 110, 32700

    models = {**BASELINES, **STATISTICAL}
    for col in train.columns:
        print(f"\n== {col} ==")
        print(compare(train[col].to_numpy(), models, HORIZON).round(4))
    print("\nfallbacks:", FALLBACKS["n"])


if __name__ == "__main__":
    main()
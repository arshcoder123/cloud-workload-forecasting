import pandas as pd

HORIZON = 24

def load (data_dir):
    cpu = pd.read_csv(f"{data_dir}/vm_cpu_readings.csv.gz")
    meta = pd.read_csv(f"{data_dir}/vm_metadata.csv.gz",
                       dtype={"virtual_core_bucket": str, "memory_gb_bucket": str})

    return cpu, meta

def build_targets(cpu):
    cpu = cpu.dropna(subset=["avg_cpu"]).drop_duplicates(["vm_id","timestamp"])
    g = cpu.groupby("timestamp")
    y = pd.DataFrame({
        "active_vm_count":g["vm_id"].nunique().astype(float),
        "aggregate_cpu_utilization": g["avg_cpu"].mean().clip(0,100)
    })

    return y.sort_index()

def split(y,h=HORIZON):
    if len(y)<= h:
        raise ValueError("series shorter than horizon")
    return y.iloc[:-h], y.index[-h:]
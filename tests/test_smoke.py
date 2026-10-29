import sys
import numpy as np
import pandas as pd
import pytest

from forecasting import predict


def make_data(path, n_vms=20, n_steps=60, start=1000, vary_count=False, seed=0):
    rng = np.random.default_rng(seed)
    ts = start + 300 * np.arange(n_steps)          # not starting at 0: nothing hard-coded
    rows = []
    for v in range(n_vms):
        for t in ts:
            if vary_count and rng.random() < 0.1:   # missing readings
                continue
            avg = float(rng.uniform(1, 20))
            rows.append((t, f"vm{v}", avg * 0.5, avg * 2, avg))
    pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]) \
        .to_csv(path / "vm_cpu_readings.csv.gz", index=False)
    pd.DataFrame({
        "vm_id": [f"vm{v}" for v in range(n_vms)],
        "subscription_id": "s", "deployment_id": "d",
        "timestamp_vm_created": 0, "timestamp_vm_deleted": 10**7,
        "max_cpu": 50.0, "avg_cpu": 10.0, "p95_max_cpu": 40.0,
        "vm_category": "Interactive",
        "virtual_core_bucket": "2", "memory_gb_bucket": "4",
    }).to_csv(path / "vm_metadata.csv.gz", index=False)
    return ts


@pytest.mark.parametrize("n_steps,vary", [(60, False), (60, True), (30, False)])
def test_predict_runs_on_tiny_data(tmp_path, monkeypatch, n_steps, vary):
    ts = make_data(tmp_path, n_steps=n_steps, vary_count=vary)
    out = tmp_path / "predictions.csv"
    monkeypatch.setattr(sys, "argv",
                        ["predict", "--data", str(tmp_path), "--output", str(out)])
    predict.main()

    d = pd.read_csv(out)
    assert list(d.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(d) == 24
    assert d.timestamp.tolist() == ts[-24:].tolist()      # taken from the data
    assert d.timestamp.is_unique and d.timestamp.is_monotonic_increasing
    assert np.isfinite(d.drop(columns="timestamp").to_numpy()).all()
    assert (d.active_vm_count >= 0).all()
    assert d.aggregate_cpu_utilization.between(0, 100).all()
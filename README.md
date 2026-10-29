# Cloud Workload Forecasting Challenge

This project forecasts the next 2 hours of cloud workload using real VM data from the Microsoft Azure Public Dataset V2 (the `data-v1` release supplied with the challenge).

It forecasts two things, for the final 24 timestamps (24 x 5 minutes = 2 hours):

- `active_vm_count`: how many VMs are running
- `aggregate_cpu_utilization`: the average CPU % over all VMs

It also turns the forecast into a simple capacity-planning rule. The full write-up is in `reports/final_report.md`.

## Results in short

- **VM count:** it is exactly 10,000 at every timestamp, so the forecast uses persistence ("the future equals the last value") and has zero error.
- **CPU:** the model chosen by validation is ARMA(1,1). In rolling-origin validation it is about 24% better than persistence (MAE 0.191 vs 0.252 CPU points).
- **Holdout:** scored once, after the model was fixed. MAE 0.099 vs 0.147 for persistence, about 32% better. This is one short window, so the validation result is the stronger evidence.
- **Honest note:** the top four models are within 0.01 MAE of each other, so the choice between them is weak evidence.
- **Capacity plan:** plan for the forecast plus a 90% safety margin (about 4.6% above the forecast), alert on the share of VMs above 80% CPU, and handle the roughly 1% of always-hot VMs one by one.

## What you need

- Python 3.11 or newer
- The challenge data in `challenge_data/` (not stored in this repository):

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Download `data-v1.tar.gz` from the `data-v1` release of the challenge repository and extract it:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

Optional check that the files are correct:

```bash
cd challenge_data
sha256sum -c SHA256SUMS
cd ..
```

## Setup

Run these from the repository root.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Generate the predictions (one command)

```bash
python -m forecasting.predict --data challenge_data --output predictions.csv
```

This prints the model chosen for each target and writes `predictions.csv`:

- exactly 24 rows
- columns: `timestamp,active_vm_count,aggregate_cpu_utilization`
- no index column
- timestamps taken from the last 24 timestamps in the data
- `active_vm_count` is never below 0, and CPU is kept within 0 to 100

Nothing is hard-coded to the supplied files. The numbers 110, 24 and 134 come from the data, so the command also works on a smaller dataset with the same columns.

## Other commands

Run the tests:

```bash
python -m pytest -q
```

Print the model comparison table (training window only):

```bash
python -m forecasting.run_validation --data challenge_data
```

Score the predictions against the holdout. Run this once, after the model is fixed, and change nothing afterwards:

```bash
python -m forecasting.evaluate_holdout --data challenge_data --predictions predictions.csv
```

## Notebooks

The analysis notebooks are in `notebooks/`. Run them from that folder, with the `.venv` as the kernel, after `predictions.csv` exists.

- the EDA notebook: understanding the data, with plots
- `02_model_comparison.ipynb`: model comparison, final forecast, safety margin, capacity rules and the one-time holdout check

Figures are saved to `reports/figures/`.

## How it works

**1. Build the targets.** Rows with a missing `avg_cpu` are dropped (not filled) and duplicate `(vm_id, timestamp)` rows are removed. For each timestamp, `active_vm_count` is the number of distinct VMs and `aggregate_cpu_utilization` is the mean `avg_cpu` across VMs, clipped to 0 to 100. This is done on the full series first, then the series is split in time: the first 110 points are for training and the last 24 are the holdout.

**2. Compare models fairly.** Rolling-origin validation: at each origin a model sees only earlier data, forecasts the next 24 steps, and is scored against what really happened. The origins move forward in time, with 7 origins on this data. Data is never shuffled.

**3. Models tried.**

- baselines: persistence, 6- and 12-step moving averages, training mean, drift
- statistical: AR(1), AR(2), ARMA(1,1), ARIMA(1,1,1), IMA(1,1), exponential smoothing (simple and damped), seasonal ARIMA
- machine learning: ridge regression on 12 lags plus a rolling mean, one model per step ahead

No neural networks were used: 110 points are too few and the series has little structure to learn.

**4. Choose the model.** The rule, fixed in advance: lowest average MAE in validation, and if all models tie, use persistence. It is applied separately to each target.

**5. Scores used.** MAE (the selection score), RMSE, MASE and the improvement over persistence. MASE is scaled by the one-step error, so values above 1 are normal for 24-step forecasts. It is undefined when the series is constant, as for the VM count.

**6. Capacity rule.** A safety margin from the validation forecast errors (90% band), an alert on the share of VMs above 80% CPU, and per-VM handling of the persistently hot VMs. Details are in the report.

## Avoiding leakage

- The final 24 values are never used to fit, tune, select a model, or build a feature.
- Each validation fit uses only data before its origin.
- These metadata columns are not used, because they cover the whole VM lifetime or reveal the future: `avg_cpu`, `max_cpu`, `p95_max_cpu` and `timestamp_vm_deleted`. `timestamp_vm_created` is always 0.
- Models use only past values of the target series.

**Disclosure:** early exploration printed descriptive statistics of the full series, including the holdout period (the holdout mean was printed once). These were never used to choose a model, a feature or a margin. Model selection used only rolling-origin validation on the first 110 timestamps with the rule above, and the holdout was scored once at the end.

## Known limitations

- The VM count never changes in this release (only VMs alive for the whole window were kept), so it says nothing about real VM arrivals and departures.
- Only about 9 hours of training data (11 in total), so daily and weekly patterns cannot be modelled.
- The safety margin and the alert level come from a few overlapping windows, so they are rough estimates.
- The CPU target is the plain average over VMs, not total compute demand.
- The holdout is a single 2-hour window.
- If a statistical model fails to fit, the code falls back to the training mean and counts it. This did not happen in the reported run.

## Project layout

```text
.
├── README.md
├── pyproject.toml
├── predictions.csv            (generated)
├── src/forecasting/
│   ├── data.py                load data, build targets, split
│   ├── models.py              baselines and forecasting models
│   ├── validate.py            metrics and rolling-origin validation
│   ├── predict.py             writes predictions.csv
│   ├── run_validation.py      prints the model comparison
│   └── evaluate_holdout.py    one-time holdout score
├── notebooks/                 EDA and model comparison
├── reports/
│   ├── final_report.md
│   └── figures/
└── tests/
```

Do not commit `challenge_data/`, `.venv/`, or notebook checkpoints. Check that `.gitignore` lists them.

## Data source

Microsoft Azure Public Dataset V2: <https://github.com/Azure/AzurePublicDataset>. The curated `data-v1` release was supplied with the challenge.
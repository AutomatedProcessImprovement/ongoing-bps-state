# Process State Evaluation

Research codebase for evaluating process-state-based short-term simulation, supporting two papers:

1. **ICPM-2025** — Three-flavour comparison (process state vs warm-up approaches)
2. **Uncertainty/clustering extension** — Confidence estimation with clustering models

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Directory Structure

```
process_state/
  main.py                              -- CLI entry point
  src/                                 -- core library (process state + simulation)
  evaluation/                          -- paper evaluation pipelines
    evaluate_with_existing_alog.py     -- three-flavour comparison (ICPM-2025)
    clustered_short_term_simulation.py -- uncertainty/clustering extension
    evaluation.py                      -- shared evaluation metrics
    helper.py                          -- data I/O and window utilities
    rtd.py                             -- remaining time distribution metric
    check_ci_calibration.py            -- CI calibration validation
    check_clusters.py                  -- visualization/analysis
    script.py                          -- post-processing helper
  tests/                               -- automated test suite (pytest)
  tools/
    fix_timestamps.py                  -- timestamp format fixer
  samples/
    icpm-2025/                         -- data for ICPM-2025 paper
    extension-uncertainty/             -- data for clustering/uncertainty extension
```

## Running Evaluation

### ICPM-2025 pipeline

```bash
python -m evaluation.evaluate_with_existing_alog DATASET_NAME --runs 10 --cut-strategy fixed
```

**Arguments:**
- `DATASET_NAME`: name of a dataset (e.g. `BPIC_2012`, `LOAN_STABLE`) or group (`ALL`, `SYNTHETIC`, `REAL-LIFE`).
- `--runs`: number of Monte-Carlo repetitions per cut-off (default: 10).
- `--cut-strategy`: method to choose cut-off timestamps.
  - `fixed` — single timestamp from dataset config.
  - `wip3` — three WiP percentiles (10%, 50%, 90%).
  - `segment10` — ten random points in equal time segments.

### Datasets

**Real-life:** BPIC_2012, BPIC_2017, WORK_ORDERS

**Synthetic:** LOAN_STABLE, LOAN_CIRCADIAN, LOAN_UNSTABLE, P2P_STABLE, P2P_CIRCADIAN, P2P_UNSTABLE

### Output

Results are saved in `outputs/<DATASET>/<run_id>/`, including:
- Reference subsets (`A_event_filter.csv`, `A_ongoing.csv`, `A_complete.csv`)
- Simulation logs and statistics for each repetition
- `final_results.json` with per-cut and overall aggregated metrics

## Running Tests

```bash
pytest tests/ -v
```

## Samples

- `samples/icpm-2025/` — Real-life and synthetic event logs for the ICPM-2025 paper
- `samples/extension-uncertainty/` — Pre-split train/test logs for the clustering extension

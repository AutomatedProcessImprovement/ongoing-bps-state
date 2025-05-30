# Process State Evaluation

This repository contains code for evaluating process state computation approaches using various real-life and synthetic event logs.

## Installation

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- log_distance_measures
- pix-framework
- ongoing_process_state
- Prosimos (custom branch for short-term simulation)

## Running Evaluation

The main evaluation script is `test/evaluate_with_existing_alog.py`. It compares different approaches for computing process state using various event logs and cut-off strategies.

### Configuration

All dataset configurations are defined in the script’s `DATASETS` section, including:
- Real-life datasets:
  - BPIC 2012
  - BPIC 2017
  - Work Orders
- Synthetic datasets:
  - Loan Application (stable, circadian, unstable)
  - P2P (stable, circadian, unstable)

You can also use dataset groups:
- `ALL` – all datasets
- `SYNTHETIC` – all synthetic datasets
- `REAL-LIFE` – all real-life datasets

### Running

To run the evaluation:

### Running

To run the evaluation:

```
python test/evaluate_with_existing_alog.py DATASET_NAME --runs 10 --cut-strategy fixed
```

**Arguments:**
- `DATASET_NAME`: name of a dataset (like `BPIC_2012`, `BPIC_2017`, etc.) or group (`ALL`, `SYNTHETIC`, `REAL-LIFE`).
- `--runs`: number of repetitions per cut-off point (default: 10).
  - For each cut-off timestamp (determined by the `--cut-strategy`), the evaluation is repeated this many times (Monte Carlo-style), ensuring robust average metrics.
- `--cut-strategy`: method to determine cut-off timestamps for evaluation.
  - `fixed`: Uses exactly one cut-off timestamp specified in the dataset’s configuration.
  - `wip3`: Generates three cut-off timestamps based on Work-in-Process (WiP) percentiles:
    - 10%, 50%, and 90% of the maximum observed WiP.
    - WiP is calculated as the number of active cases at each event arrival.
  - `segment10`: Generates ten cut-off timestamps by:
    - Skipping the first and last `horizon` (to avoid boundary effects).
    - Dividing the remaining interval into ten equal segments.
    - Picking a random timestamp from within each segment.

**Example usage:**

To run the evaluation for the `LOAN_STABLE` dataset with 5 repetitions for each of 10 random cut-offs:

```
python test/evaluate_with_existing_alog.py LOAN_STABLE --runs 5 --cut-strategy segment10
```

---

### Output

Results are saved in the `outputs/<DATASET>/<run_id>` directory, including:
- Reference subsets (`A_event_filter.csv`, `A_ongoing.csv`, `A_complete.csv`)
- Simulation logs and statistics for each repetition
- Aggregated metrics for each cut-off
- Final report: `final_results.json` containing:
  - Number of runs
  - Cut strategy
  - Per cut-off aggregated metrics
  - Overall averages and comparisons between flavours

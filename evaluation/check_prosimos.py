from pathlib import Path
import pandas as pd
from prosimos.simulation_engine import run_simulation
import json
from datetime import datetime, timezone, timedelta


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "samples" / "icpm-2025" / "synthetic"

CUT = "2025-01-20T10:00:00Z"
TOTAL_CASES = 5_000

OUT_DIR = ROOT / "tmp_dupe_check"
OUT_DIR.mkdir(exist_ok=True)

OUT_LOG  = OUT_DIR / "loan_stable_sim_log.csv"
OUT_STAT = OUT_DIR / "loan_stable_sim_stats.csv"

CUT_DT = datetime.fromisoformat(CUT.replace("Z", "+00:00"))
HORIZON_DT = CUT_DT + timedelta(days=1)

with open("outputs/LOAN_STABLE/jhd9jn/2025-01-13T19-16-19.260000+00-00/2/process_state.json", "r") as f:
    process_state = json.load(f)

run_simulation(
    bpmn_path="samples/icpm-2025/synthetic/Loan-stable.bpmn",
    json_path="samples/icpm-2025/synthetic/Loan-stable.json",
    total_cases=TOTAL_CASES,
    stat_out_path="outputs/out_stats.csv",
    log_out_path="outputs/out_log.csv",
    starting_at=CUT,
    process_state=process_state,
    simulation_horizon=HORIZON_DT,   # or HORIZON_DT.isoformat()
)

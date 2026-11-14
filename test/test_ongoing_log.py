import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.compute_bps_state_and_run_simulation import compute_bps_state_and_run_simulation, ongoing_case_ids


class LogThatEndsAtTheStart(unittest.TestCase):
    sample = Path(__file__).resolve().parent.parent / "samples" / "dev-samples"
    start = "2025-03-10T09:00:00+02:00"
    horizon = "2025-03-11T09:00:00+02:00"

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.folder.name)
        self.path = Path(self.folder.name)

    def open_cases(self, log_path, name):
        state = self.path / f"{name}.pkl"
        compute_bps_state_and_run_simulation(
            ongoing_log_path=log_path,
            bpmn_model_path=self.sample / "Loan_Application.bpmn",
            bpmn_parameters_path=self.sample / "Loan_Application.json",
            start_time=self.start,
            simulation_horizon=self.horizon,
            short_term_simulated_log_path=self.path / f"{name}.csv",
            process_state_path=state,
            seed=7,
        )
        return set(ongoing_case_ids(state))

    def test_cases_waiting_between_activities_stay_open(self):
        log = pd.read_csv(self.sample / "Loan_Application_log.csv", dtype=str)
        start = pd.Timestamp(self.start)
        log = log[pd.to_datetime(log["StartTime"], utc=True) <= start]
        running = pd.to_datetime(log["EndTime"], utc=True) > start
        log.loc[running, "EndTime"] = ""
        recorded = self.path / "recorded.csv"
        log.to_csv(recorded, index=False)

        expected = self.open_cases(self.sample / "Loan_Application_log.csv", "full")
        self.assertGreater(len(expected), running.sum())
        self.assertEqual(self.open_cases(recorded, "recorded"), expected)


if __name__ == "__main__":
    unittest.main()

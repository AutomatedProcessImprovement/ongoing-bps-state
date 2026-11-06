import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.compute_bps_state_and_run_simulation import compute_bps_state_and_run_simulation, rerun_from_process_state
from src.compute_frontend_events_from_trace import sim_log_ids
from src.run_metrics import paired_change, resource_timeline, summarize_log
from src.scenario import InvalidEdit, apply_edits, expand_amounts, pool_members, pool_size

CALENDAR = {
    "id": "weekdays",
    "name": "weekdays",
    "time_periods": [{"from": "MONDAY", "to": "FRIDAY", "beginTime": "09:00:00.000", "endTime": "17:00:00.000"}],
}


def resource(resource_id, amount=1):
    return {"id": resource_id, "name": resource_id, "amount": amount, "cost_per_hour": 1, "calendar": "weekdays"}


def assignment(resource_id, mean=600):
    return {
        "resource_id": resource_id,
        "distribution_name": "gamma",
        "distribution_params": [{"value": mean}, {"value": 400}, {"value": 10}, {"value": 2000}],
    }


def parameters():
    return {
        "resource_calendars": [CALENDAR],
        "arrival_time_distribution": {"distribution_name": "fix", "distribution_params": [{"value": 900}]},
        "resource_profiles": [
            {"id": "clerks", "name": "Clerks", "resource_list": [resource("clerk", 3)]},
            {"id": "officers", "name": "Officers", "resource_list": [resource("a"), resource("b")]},
        ],
        "task_resource_distribution": [
            {"task_id": "check", "resources": [assignment("clerk")]},
            {"task_id": "approve", "resources": [assignment("a"), assignment("b")]},
            {"task_id": "sign", "resources": [assignment("b")]},
        ],
    }


def task(parameters_, task_id):
    return next(item for item in parameters_["task_resource_distribution"] if item["task_id"] == task_id)


class ApplyEdits(unittest.TestCase):
    def test_no_edits_leave_the_parameters_alone(self):
        self.assertEqual(apply_edits(parameters(), {}), parameters())

    def test_adding_to_a_pool_with_an_amount_keeps_the_simulator_names(self):
        edited = apply_edits(parameters(), {"pools": {"clerks": 5}})
        pool = edited["resource_profiles"][0]
        self.assertEqual(pool_size(pool), 5)
        self.assertEqual([member["name"] for member in pool_members(pool)][:3], ["clerk_0", "clerk_1", "clerk_2"])
        self.assertEqual(
            {item["resource_id"] for item in task(edited, "check")["resources"]},
            {resource_["id"] for resource_ in pool["resource_list"]},
        )

    def test_removing_keeps_a_resource_for_every_task(self):
        edited = apply_edits(parameters(), {"pools": {"officers": 1}})
        self.assertEqual([r["id"] for r in edited["resource_profiles"][1]["resource_list"]], ["b"])
        self.assertEqual([a["resource_id"] for a in task(edited, "approve")["resources"]], ["b"])
        self.assertEqual([a["resource_id"] for a in task(edited, "sign")["resources"]], ["b"])

    def test_a_pool_cannot_lose_its_only_performer(self):
        with self.assertRaises(InvalidEdit):
            apply_edits(parameters(), {"pools": {"officers": 0}})

    def test_more_arrivals_mean_shorter_gaps(self):
        edited = apply_edits(parameters(), {"arrival_percent": 50})
        self.assertAlmostEqual(edited["arrival_time_distribution"]["distribution_params"][0]["value"], 600)

    def test_task_durations_scale_the_mean_and_square_the_variance(self):
        edited = apply_edits(parameters(), {"tasks": {"check": -50}})
        values = [p["value"] for p in task(edited, "check")["resources"][0]["distribution_params"]]
        self.assertEqual(values, [300, 100, 5, 1000])
        self.assertEqual(task(edited, "check")["resources"][0]["resource_id"], "clerk")

    def test_unknown_targets_and_extreme_values_are_refused(self):
        for edits in [{"pools": {"nope": 2}}, {"tasks": {"nope": 10}}, {"arrival_percent": -95},
                      {"tasks": {"check": 400}}, {"pools": {"clerks": 0}}]:
            with self.assertRaises(InvalidEdit):
                apply_edits(parameters(), edits)

    def test_fuzzy_models_cannot_resize_pools(self):
        with self.assertRaises(InvalidEdit):
            apply_edits({**parameters(), "model_type": "FUZZY"}, {"pools": {"clerks": 4}})


def log_rows(*rows):
    log = pd.DataFrame(rows, columns=[
        sim_log_ids.case, sim_log_ids.activity, sim_log_ids.enabled_time, sim_log_ids.start_time,
        sim_log_ids.end_time, sim_log_ids.resource,
    ])
    for column in [sim_log_ids.enabled_time, sim_log_ids.start_time, sim_log_ids.end_time]:
        log[column] = pd.to_datetime(log[column], utc=True)
    return log


class SummarizeLog(unittest.TestCase):
    start = pd.Timestamp("2025-03-03T00:00:00Z")
    end = pd.Timestamp("2025-03-03T10:00:00Z")

    def test_waiting_counts_work_that_only_starts_after_the_horizon(self):
        summary = summarize_log(log_rows(
            (1, "A", "2025-03-03T01:00Z", "2025-03-03T03:00Z", "2025-03-03T04:00Z", "x"),
            (2, "A", "2025-03-03T08:00Z", "2025-03-03T12:00Z", "2025-03-03T13:00Z", "x"),
        ), self.start, self.end, {})
        self.assertEqual(summary["waitMs"], 3 * 3600000)
        self.assertEqual(summary["waitingAtEnd"], 1)
        self.assertEqual(summary["finished"], 1)
        self.assertEqual(summary["openAtEnd"], 1)
        self.assertAlmostEqual(summary["queueAverage"], (2 + 2) / 10)
        self.assertEqual(summary["queuePeak"], 1)
        self.assertEqual(summary["tasks"]["A"]["count"], 2)

    def test_work_in_progress_at_the_start_is_not_waiting(self):
        summary = summarize_log(log_rows(
            (1, "A", "2025-03-02T20:00Z", "2025-03-02T22:00Z", "2025-03-03T01:00Z", "x"),
        ), self.start, self.end, {"1": "2025-03-02T18:00:00+00:00"})
        self.assertIsNone(summary["waitMs"])
        self.assertEqual(summary["queueAverage"], 0)
        self.assertEqual(summary["arrived"], 0)
        self.assertEqual(summary["cycleMs"], 7 * 3600000)

    def test_paired_change_reports_the_spread_over_seeds(self):
        def side(value):
            return {"waitMs": value, "queueAverage": value, "cycleMs": None, "finished": value,
                    "tasks": {"A": {"waitMs": value, "queueAverage": value}}}
        change = paired_change([{"asIs": side(10), "toBe": side(8)}, {"asIs": side(10), "toBe": side(14)}])
        self.assertEqual(change["seeds"], 2)
        self.assertEqual(change["changes"]["waitMs"], {"mean": 1, "low": -2, "high": 4})
        self.assertIsNone(change["changes"]["cycleMs"]["mean"])
        self.assertEqual(change["tasks"]["A"]["queueAverage"], {"mean": 1, "low": -2, "high": 4})


class ResourceTimeline(unittest.TestCase):
    def test_counts_busy_idle_and_off_shift_in_simulator_time(self):
        params = {
            "resource_calendars": [CALENDAR],
            "resource_profiles": [{"id": "p", "name": "P", "resource_list": [resource("r", 2)]}],
        }
        start = pd.Timestamp("2025-03-03T08:00:00+02:00")
        end = pd.Timestamp("2025-03-03T18:00:00+02:00")
        timeline = resource_timeline(log_rows(
            ("1", "A", "2025-03-03T07:00Z", "2025-03-03T07:00Z", "2025-03-03T09:00Z", "r_0"),
            ("2", "A", "2025-03-03T08:00Z", "2025-03-03T08:00Z", "2025-03-03T10:00Z", "unknown"),
        ), params, start, end)
        pool = timeline["pools"][0]
        hours = timeline["stepMs"] / 3600000
        self.assertAlmostEqual(hours, 1)
        self.assertEqual(pool["busy"][:4], [0, 1, 1, 0])
        self.assertEqual(pool["idle"][:4], [0, 1, 1, 2])
        self.assertEqual(pool["busy"][10], 0)
        self.assertEqual(pool["idle"][10], 0)
        self.assertAlmostEqual(pool["utilization"], 2 / 16, places=2)
        self.assertEqual(sorted(r["name"] for r in pool["resources"]), ["r_0", "r_1"])


class RerunFromState(unittest.TestCase):
    sample = Path(__file__).resolve().parent.parent / "samples" / "dev-samples"
    start = "2025-03-10T09:00:00+02:00"
    horizon = "2025-03-17T09:00:00+02:00"

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.folder.name)
        self.path = Path(self.folder.name)
        self.state = self.path / "state.pkl"
        self.first_log = self.path / "first.csv"
        self.frame, _ = compute_bps_state_and_run_simulation(
            ongoing_log_path=self.sample / "Loan_Application_log.csv",
            bpmn_model_path=self.sample / "Loan_Application.bpmn",
            bpmn_parameters_path=self.sample / "Loan_Application.json",
            start_time=self.start,
            simulation_horizon=self.horizon,
            short_term_simulated_log_path=self.first_log,
            process_state_path=self.state,
            seed=7,
        )

    def rerun(self, parameters_path, name, seed=7):
        log_path = self.path / name
        frame = rerun_from_process_state(
            process_state_path=self.state,
            bpmn_model_path=self.sample / "Loan_Application.bpmn",
            bpmn_parameters_path=parameters_path,
            start_time=self.start,
            simulation_horizon=self.horizon,
            short_term_simulated_log_path=log_path,
            seed=seed,
        )
        return frame, log_path

    def test_same_state_and_seed_reproduce_the_run_exactly(self):
        frame, log_path = self.rerun(self.sample / "Loan_Application.json", "same.csv")
        self.assertEqual(frame, self.frame)
        self.assertEqual(log_path.read_bytes(), self.first_log.read_bytes())

    def test_another_seed_gives_another_run_from_the_same_state(self):
        frame, log_path = self.rerun(self.sample / "Loan_Application.json", "other.csv", seed=8)
        self.assertEqual(frame, self.frame)
        self.assertNotEqual(log_path.read_bytes(), self.first_log.read_bytes())

    def test_listing_a_pool_one_by_one_does_not_change_the_run(self):
        original = json.loads((self.sample / "Loan_Application.json").read_text())
        expanded = copy.deepcopy(original)
        for pool in expanded["resource_profiles"]:
            expand_amounts(expanded, pool)
        path = self.path / "expanded.json"
        path.write_text(json.dumps(expanded))
        _, log_path = self.rerun(path, "expanded.csv")
        self.assertEqual(log_path.read_bytes(), self.first_log.read_bytes())

    def test_a_backlog_does_not_lose_arrivals(self):
        original = json.loads((self.sample / "Loan_Application.json").read_text())
        pool = max(original["resource_profiles"], key=pool_size)
        path = self.path / "smaller.json"
        path.write_text(json.dumps(apply_edits(original, {"pools": {pool["id"]: 1}})))
        _, log_path = self.rerun(path, "smaller.csv")
        cases = pd.read_csv(self.first_log)["case_id"].nunique()
        self.assertEqual(pd.read_csv(log_path)["case_id"].nunique(), cases)


if __name__ == "__main__":
    unittest.main()

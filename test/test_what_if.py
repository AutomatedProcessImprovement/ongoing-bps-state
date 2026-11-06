import copy
import json
import multiprocessing
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from main_web_app import compute_ongoing_task_times
from src.compute_bps_state_and_run_simulation import compute_bps_state_and_run_simulation, \
    generate_events_with_token_movements, ongoing_case_ids, rerun_from_process_state, write_empty_process_state
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

    def test_a_case_already_queueing_at_the_start_counts_with_its_whole_wait(self):
        summary = summarize_log(log_rows(
            (1, "A", "2025-03-02T22:00Z", "2025-03-03T02:00Z", "2025-03-03T03:00Z", "x"),
        ), self.start, self.end, {"1": "2025-03-02T18:00:00+00:00"})
        self.assertEqual(summary["waitMs"], 4 * 3600000)
        self.assertAlmostEqual(summary["queueAverage"], 2 / 10)

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


class WorkerProcesses(unittest.TestCase):
    def test_workers_do_not_fork_from_the_threaded_server(self):
        self.assertEqual(multiprocessing.get_start_method(), "forkserver")


class OngoingTaskTimes(unittest.TestCase):
    def task_times(self, *rows):
        with tempfile.TemporaryDirectory() as folder:
            log_path = Path(folder) / "log.csv"
            log_rows(*rows).to_csv(log_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
            bpmn = SimpleNamespace(nodes=[SimpleNamespace(name="A", id="Task_A"), SimpleNamespace(name="B", id="Task_B")])
            with patch("main_web_app.read_bpmn_model", return_value=bpmn):
                return compute_ongoing_task_times(log_path, Path("model.bpmn"), pd.Timestamp("2025-03-03T00:00:00Z"), [1, "2"])

    def test_keeps_the_times_of_tasks_open_at_the_start(self):
        self.assertEqual(self.task_times(
            (1, "A", "2025-03-02T20:00Z", "2025-03-03T02:00Z", "2025-03-03T03:00Z", "x"),
            (2, "B", "2025-03-02T20:00Z", "2025-03-02T22:00Z", "2025-03-03T03:00Z", "x"),
        ), {
            "1": {"Task_A": {"enabled": "2025-03-02T20:00:00.000000+0000", "started": None}},
            "2": {"Task_B": {
                "enabled": "2025-03-02T20:00:00.000000+0000",
                "started": "2025-03-02T22:00:00.000000+0000",
            }},
        })

    def test_skips_later_work_other_cases_and_unknown_activities(self):
        self.assertEqual(self.task_times(
            (1, "A", "2025-03-03T01:00Z", "2025-03-03T02:00Z", "2025-03-03T03:00Z", "x"),
            (3, "A", "2025-03-02T20:00Z", "2025-03-03T02:00Z", "2025-03-03T03:00Z", "x"),
            (1, "Gone", "2025-03-02T20:00Z", "2025-03-03T02:00Z", "2025-03-03T03:00Z", "x"),
        ), {})


class TokenMovements(unittest.TestCase):
    def replayed_activities(self, *rows):
        start = pd.Timestamp("2025-03-03T00:00:00Z")
        frame = [{"case_id": 1, "active_elements": {"token_0": "flow"}}]
        with tempfile.TemporaryDirectory() as folder:
            log_path = Path(folder) / "log.csv"
            log_rows(*rows).to_csv(log_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
            graph_path = Path(folder) / "graph.tgf"
            graph_path.touch()
            with patch("src.compute_bps_state_and_run_simulation.read_bpmn_model"), \
                    patch("src.compute_bps_state_and_run_simulation.read_reachability_graph"), \
                    patch("src.compute_bps_state_and_run_simulation.compute_token_movements", return_value=[]) as replay:
                generate_events_with_token_movements(Path("model.bpmn"), start, log_path, graph_path, frame)
        return list(replay.call_args.kwargs["trace"][sim_log_ids.activity])

    def test_work_with_no_duration_at_the_start_is_replayed(self):
        self.assertEqual(self.replayed_activities(
            (1, "done", "2025-03-02T18:00Z", "2025-03-02T19:00Z", "2025-03-02T20:00Z", "x"),
            (1, "instant", "2025-03-02T23:00Z", "2025-03-03T00:00Z", "2025-03-03T00:00Z", "x"),
            (1, "next", "2025-03-03T00:00Z", "2025-03-03T01:00Z", "2025-03-03T02:00Z", "x"),
        ), ["instant", "next"])

    def test_work_finished_before_the_start_is_left_out(self):
        self.assertEqual(self.replayed_activities(
            (1, "done", "2025-03-02T18:00Z", "2025-03-02T19:00Z", "2025-03-02T20:00Z", "x"),
            (1, "running", "2025-03-02T21:00Z", "2025-03-02T22:00Z", "2025-03-03T02:00Z", "x"),
        ), ["running"])


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


    def test_arrivals_between_the_last_known_arrival_and_the_start_do_not_pile_up(self):
        with open(self.state, "rb") as state_file:
            saved_state = pickle.load(state_file)
        saved_state["partial_state"]["last_case_arrival"] = pd.Timestamp(self.start) - pd.Timedelta(days=7)
        with open(self.state, "wb") as state_file:
            pickle.dump(saved_state, state_file)

        frame, log_path = self.rerun(self.sample / "Loan_Application.json", "gap.csv")

        log = pd.read_csv(log_path)
        ongoing = {str(case["case_id"]) for case in frame}
        new_cases = log[~log["case_id"].astype(str).isin(ongoing)]
        arrivals = pd.to_datetime(new_cases["enable_time"], utc=True, format="ISO8601").groupby(new_cases["case_id"]).min()
        at_the_start = (arrivals <= pd.Timestamp(self.start) + pd.Timedelta(seconds=1)).sum()
        self.assertGreater(len(arrivals), 0)
        self.assertLessEqual(at_the_start, 1)

    def test_an_empty_start_keeps_no_ongoing_case(self):
        empty_state = self.path / "empty.pkl"
        write_empty_process_state(self.state, empty_state)
        self.assertTrue(ongoing_case_ids(self.state))
        self.assertEqual(ongoing_case_ids(empty_state), [])

        log_path = self.path / "empty.csv"
        frame = rerun_from_process_state(
            process_state_path=empty_state,
            bpmn_model_path=self.sample / "Loan_Application.bpmn",
            bpmn_parameters_path=self.sample / "Loan_Application.json",
            start_time=self.start,
            simulation_horizon=self.horizon,
            short_term_simulated_log_path=log_path,
            seed=7,
        )
        self.assertEqual(frame, [])
        log = pd.read_csv(log_path)
        self.assertGreater(log["case_id"].nunique(), 0)
        self.assertTrue((pd.to_datetime(log["start_time"], utc=True) >= pd.Timestamp(self.start)).all())


if __name__ == "__main__":
    unittest.main()

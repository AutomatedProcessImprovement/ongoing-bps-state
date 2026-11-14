import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.log_input import InvalidLog, prepare_event_log

MAPPING = {"case": "CaseId", "activity": "Activity", "start": "StartTime", "end": "EndTime", "resource": "Resource",
           "enablement": "__DISCOVER__"}
HEADER = "CaseId,Activity,StartTime,EndTime,Resource\n"
ROWS = ("1,Check,2025-03-25 09:00:00+02:00,2025-03-25 09:30:00+02:00,Ann\n"
        "2,Approve,2025-03-25 10:00:00+02:00,2025-03-25 11:00:00+02:00,Bob\n")


class PrepareEventLog(unittest.TestCase):
    def prepare(self, content, mapping=MAPPING):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log.csv"
            path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
            renumbered = prepare_event_log(path, mapping, "__DISCOVER__")
            return pd.read_csv(path, dtype=str), renumbered

    def assertRejected(self, content, *fragments, mapping=MAPPING):
        with self.assertRaises(InvalidLog) as raised:
            self.prepare(content, mapping)
        for fragment in fragments:
            self.assertIn(fragment, str(raised.exception))

    def test_a_valid_log_is_kept(self):
        frame, renumbered = self.prepare(HEADER + ROWS)
        self.assertEqual(list(frame["CaseId"]), ["1", "2"])
        self.assertEqual(frame.at[0, "StartTime"], "2025-03-25 09:00:00+02:00")
        self.assertEqual(renumbered, {})

    def test_empty_files_and_headers_without_events_are_rejected(self):
        self.assertRejected("", "empty")
        self.assertRejected(HEADER, "no events")

    def test_binary_files_are_rejected(self):
        self.assertRejected(b"PK\x03\x04\x00\x00binary", "not a text CSV file")

    def test_a_single_column_is_rejected(self):
        self.assertRejected("CaseId\n1\n2\n", "single column")

    def test_semicolons_and_tabs_are_read(self):
        for delimiter in (";", "\t"):
            frame, _ = self.prepare((HEADER + ROWS).replace(",", delimiter))
            self.assertEqual(list(frame.columns), ["CaseId", "Activity", "StartTime", "EndTime", "Resource"])

    def test_latin1_files_are_read(self):
        frame, _ = self.prepare((HEADER + ROWS.replace("Ann", "Zoé")).encode("cp1252"))
        self.assertEqual(frame.at[0, "Resource"], "Zoé")

    def test_missing_mapped_columns_are_named(self):
        self.assertRejected(HEADER.replace("Resource", "Who") + ROWS, 'Resource ("Resource")', "Its columns are")

    def test_unreadable_timestamps_are_shown_with_their_row(self):
        self.assertRejected(HEADER + ROWS.replace("2025-03-25 10:00:00+02:00", "25/03/2025 10:00"),
                            'Column "StartTime"', "row 3", '"25/03/2025 10:00"', "ISO 8601")

    def test_empty_end_times_mark_activities_in_progress(self):
        frame, _ = self.prepare(HEADER + ROWS.replace("2025-03-25 11:00:00+02:00", ""))
        self.assertTrue(pd.isna(frame.at[1, "EndTime"]))
        self.assertEqual(frame.at[0, "EndTime"], "2025-03-25 09:30:00+02:00")
        self.assertRejected(HEADER + ROWS.replace("2025-03-25 10:00:00+02:00", ""), "no start timestamp", "row 3")
        self.assertRejected(HEADER + ROWS.replace("2025-03-25 11:00:00+02:00", "").replace("2025-03-25 09:30:00+02:00", ""),
                            "no end timestamp")

    def test_timestamps_without_time_zone_are_read_as_utc(self):
        frame, _ = self.prepare(HEADER + ROWS.replace("+02:00", ""))
        self.assertEqual(frame.at[0, "StartTime"], "2025-03-25 09:00:00+00:00")

    def test_timestamps_are_written_in_one_form(self):
        rows = ROWS.replace("2025-03-25 09:00:00+02:00", "2025-03-25T09:00+0200").replace(
            "2025-03-25 09:30:00+02:00", "2025-03-25 09:30:00 +02:00")
        frame, _ = self.prepare(HEADER + rows)
        self.assertEqual(frame.at[0, "StartTime"], "2025-03-25 09:00:00+02:00")
        self.assertEqual(frame.at[0, "EndTime"], "2025-03-25 09:30:00+02:00")
        frame, _ = self.prepare(HEADER + ROWS.replace("+02:00", "Z"))
        self.assertEqual(frame.at[1, "EndTime"], "2025-03-25 11:00:00+00:00")

    def test_other_timestamp_columns_get_the_same_time_zone(self):
        header = HEADER.replace("Resource\n", "Resource,enabled_time\n")
        rows = "".join(row + ",2025-03-25 08:00:00\n" for row in ROWS.replace("+02:00", "").splitlines())
        frame, _ = self.prepare(header + rows)
        self.assertEqual(frame.at[0, "enabled_time"], "2025-03-25 08:00:00+00:00")

    def test_mixing_time_zones_and_none_is_rejected(self):
        self.assertRejected(HEADER + ROWS.replace("10:00:00+02:00", "10:00:00"), "mixes timestamps")

    def test_events_that_end_before_they_start_are_rejected(self):
        self.assertRejected(HEADER + ROWS.replace("2025-03-25 11:00:00+02:00", "2025-03-25 08:00:00+02:00"),
                            "1 event in the log ends before it starts", "row 3")

    def test_empty_activities_are_rejected_and_empty_resources_filled(self):
        self.assertRejected(HEADER + ROWS.replace("Approve", ""), "no activity", "row 3")
        frame, _ = self.prepare(HEADER + ROWS.replace("Bob", ""))
        self.assertEqual(frame.at[1, "Resource"], "Unknown resource")

    def test_text_case_ids_are_renumbered(self):
        frame, renumbered = self.prepare((HEADER + ROWS).replace("\n1,", "\nA-1,").replace("\n2,", "\nB 2,"))
        self.assertEqual(list(frame["CaseId"]), ["0", "1"])
        self.assertEqual(renumbered, {"0": "A-1", "1": "B 2"})

    def test_quoted_values_with_commas_are_kept(self):
        frame, _ = self.prepare(HEADER + ROWS.replace("Check", '"Check, again"'))
        self.assertEqual(frame.at[0, "Activity"], "Check, again")


if __name__ == "__main__":
    unittest.main()

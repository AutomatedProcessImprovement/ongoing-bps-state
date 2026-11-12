import csv
import io
import re
from pathlib import Path

import pandas as pd

DELIMITERS = [",", ";", "\t", "|"]
ISO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?\s*(Z|[+-]\d{2}:?\d{2})?$")
HAS_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
TIMESTAMP_HINT = "Use ISO 8601 timestamps such as 2025-03-25 10:00:00 or 2025-03-25T10:00:00+02:00."
LABELS = {"case": "Case ID", "activity": "Activity", "start": "Start timestamp", "end": "End timestamp",
          "resource": "Resource", "enablement": "Enablement timestamp"}


class InvalidLog(ValueError):
    pass


def _decode(raw: bytes) -> str:
    if b"\x00" in raw[:4096]:
        raise InvalidLog("The log is not a text CSV file. If it is an Excel workbook, save it as CSV first.")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _delimiter(header: str) -> str:
    counts = {delimiter: header.count(delimiter) for delimiter in DELIMITERS}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise InvalidLog("The log has a single column. Its columns must be separated by commas or semicolons.")
    return best


def _row(index: int) -> int:
    return index + 2


def _example(frame: pd.DataFrame, mask: pd.Series, column: str, case_column: str) -> str:
    index = mask[mask].index[0]
    value = frame.at[index, column]
    shown = "empty" if pd.isna(value) or value == "" else f'"{value}"'
    return f"row {_row(index)}, case {frame.at[index, case_column]}: {shown}"


def _count(mask: pd.Series, noun: str) -> str:
    total = int(mask.sum())
    return f"{total} {noun if total == 1 else noun + 's'}"


def _canonical(values: pd.Series) -> pd.Series:
    return (values
            .str.replace(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?=$|[^:\d])", r"\1 \2:00", regex=True)
            .str.replace(r"\s+(?=(Z|[+-]\d{2}:?\d{2})$)", "", regex=True)
            .str.replace(r"Z$", "+00:00", regex=True)
            .str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True))


def _normalize_column(frame: pd.DataFrame, column: str):
    present = frame[column].fillna("").str.strip() != ""
    values = _canonical(frame.loc[present, column].str.strip())
    if not values.str.contains(HAS_OFFSET).any():
        values = values + "+00:00"
    frame.loc[present, column] = values
    frame.loc[~present, column] = pd.NA


def _check_timestamps(frame: pd.DataFrame, column: str, label: str, case_column: str) -> pd.Series:
    values = frame[column].fillna("").str.strip()
    empty = values == ""
    if empty.any():
        raise InvalidLog(f'{_count(empty, "row")} of the log {"has" if empty.sum() == 1 else "have"} no {label.lower()} '
                         f'in column "{column}" ({_example(frame, empty, column, case_column)}). Every event needs one.')
    unreadable = ~values.str.match(ISO_TIMESTAMP)
    if unreadable.any():
        raise InvalidLog(f'Column "{column}" ({label}) has values that are not timestamps '
                         f'({_example(frame, unreadable, column, case_column)}). {TIMESTAMP_HINT}')
    with_offset = values.str.contains(HAS_OFFSET)
    if with_offset.any() and not with_offset.all():
        raise InvalidLog(f'Column "{column}" mixes timestamps with and without a time zone '
                         f'({_example(frame, ~with_offset, column, case_column)}). Give all of them a time zone or none.')
    values = _canonical(values)
    if not with_offset.any():
        values = values + "+00:00"
    frame[column] = values
    try:
        return pd.to_datetime(values, utc=True, format="ISO8601")
    except (ValueError, TypeError) as error:
        raise InvalidLog(f'Column "{column}" ({label}) has a timestamp that does not exist: {error}. {TIMESTAMP_HINT}')


def prepare_event_log(log_path: Path, column_mapping: dict, discover_enablement: str) -> dict:
    raw = log_path.read_bytes()
    if not raw.strip():
        raise InvalidLog("The log file is empty.")
    text = _decode(raw)
    lines = text.splitlines()
    delimiter = _delimiter(lines[0])
    try:
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False, skipinitialspace=True)
    except (pd.errors.ParserError, csv.Error) as error:
        raise InvalidLog(f"The log is not a well-formed CSV file: {error}")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.replace("", pd.NA).dropna(how="all")
    if frame.empty:
        raise InvalidLog("The log has a header row but no events.")

    fields = ["case", "activity", "start", "end", "resource"]
    if column_mapping.get("enablement") not in (None, "", discover_enablement):
        fields.append("enablement")
    missing = [field for field in fields if column_mapping.get(field) not in frame.columns]
    if missing:
        names = ", ".join(f'{LABELS[field]} ("{column_mapping.get(field)}")' for field in missing)
        raise InvalidLog(f"The log has no column for {names}. Its columns are: {', '.join(frame.columns[:12])}.")

    case_column = column_mapping["case"]
    for field in ("case", "activity"):
        column = column_mapping[field]
        empty = frame[column].isna() | (frame[column].str.strip() == "")
        if empty.any():
            raise InvalidLog(f'{_count(empty, "row")} of the log {"has" if empty.sum() == 1 else "have"} no '
                             f'{LABELS[field].lower()} in column "{column}" (row {_row(empty[empty].index[0])}).')
    frame[column_mapping["resource"]] = frame[column_mapping["resource"]].fillna("Unknown resource")

    times = {field: _check_timestamps(frame, column_mapping[field], LABELS[field], case_column)
             for field in fields if field in ("start", "end", "enablement")}
    mapped = {column_mapping[field] for field in fields}
    for column in frame.columns.difference(list(mapped)):
        values = frame[column].dropna().str.strip()
        offsets = values.str.contains(HAS_OFFSET)
        if len(values) and values.str.match(ISO_TIMESTAMP).all() and (offsets.all() or not offsets.any()):
            _normalize_column(frame, column)
    backwards = times["end"] < times["start"]
    if backwards.any():
        one = backwards.sum() == 1
        raise InvalidLog(f'{_count(backwards, "event")} in the log {"ends before it starts" if one else "end before they start"} '
                         f'({_example(frame, backwards, column_mapping["end"], case_column)}). '
                         f'Check that the start and end columns are mapped the right way round.')

    case_ids = {}
    if not frame[case_column].str.fullmatch(r"\d+").all():
        codes, originals = pd.factorize(frame[case_column])
        case_ids = {str(code): original for code, original in enumerate(originals)}
        frame[case_column] = codes.astype(str)

    frame.to_csv(log_path, index=False)
    return case_ids

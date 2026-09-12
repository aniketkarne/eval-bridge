from pathlib import Path

from eval_bridge.incident import next_incident_id, COUNTER_FILE


def test_first_id_in_category(tmp_path: Path):
    (tmp_path / ".eval-bridge-counter").write_text('{"pii-leak": 0}\n')
    assert next_incident_id(tmp_path, "pii-leak") == "pii-leak-001"


def test_increments_persistently(tmp_path: Path):
    (tmp_path / ".eval-bridge-counter").write_text('{"pii-leak": 41}\n')
    assert next_incident_id(tmp_path, "pii-leak") == "pii-leak-042"


def test_unknown_category_starts_at_1(tmp_path: Path):
    assert next_incident_id(tmp_path, "tool-misuse") == "tool-misuse-001"


def test_counter_file_format(tmp_path: Path):
    assert COUNTER_FILE.name == ".eval-bridge-counter"
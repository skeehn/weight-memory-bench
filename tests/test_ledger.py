import json

import pytest

from harness import ledger


def good_row(**overrides):
    row = {
        "reader_model": "Qwen/Qwen3-0.6B",
        "reader_revision": "main",
        "tokenizer_fingerprint": "deadbeefdeadbeef",
        "arm": "full_context",
        "split": "dev",
        "seed": 1337,
        "corpus_hash": "abc123",
        "accuracy_over_all": 0.42,
    }
    row.update(overrides)
    return row


class TestAppend:
    def test_writes_a_complete_row(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        written = ledger.append(good_row(), path)
        assert written["timestamp_utc"]
        assert ledger.rows(path) == [written]

    def test_refuses_a_row_missing_provenance(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        with pytest.raises(ledger.ProvenanceIncomplete) as exc:
            ledger.append(good_row(corpus_hash=None), path)
        assert "corpus_hash" in str(exc.value)
        assert not path.exists(), "no file should be created for a rejected row"

    def test_refuses_a_row_with_empty_string_provenance(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        with pytest.raises(ledger.ProvenanceIncomplete):
            ledger.append(good_row(arm=""), path)

    def test_is_append_only(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger.append(good_row(arm="full_context"), path)
        ledger.append(good_row(arm="weight_memory"), path)
        assert [r["arm"] for r in ledger.rows(path)] == ["full_context", "weight_memory"]


class TestReadIsLineByLine:
    def test_a_corrupt_line_costs_one_row_not_the_file(self, tmp_path):
        # This is the regression that matters. A whole-file json.loads dies on line 2 with
        # "Extra data", reports the file unreadable, and inspects zero rows -- exempting
        # the exact artifact the audit exists to police.
        path = tmp_path / "ledger.jsonl"
        ledger.append(good_row(arm="a"), path)
        with open(path, "a") as fh:
            fh.write("{not valid json\n")
        ledger.append(good_row(arm="c"), path)

        items = list(ledger.read(path))
        assert len(items) == 3
        assert [type(i).__name__ for i in items] == ["dict", "BadLine", "dict"]
        assert ledger.rows(path) == [i for i in items if isinstance(i, dict)]
        assert [r["arm"] for r in ledger.rows(path)] == ["a", "c"]

    def test_whole_file_json_loads_would_fail_here(self, tmp_path):
        # Pins the premise: the file is genuinely JSON lines, so the naive read is wrong.
        path = tmp_path / "ledger.jsonl"
        ledger.append(good_row(arm="a"), path)
        ledger.append(good_row(arm="b"), path)
        with pytest.raises(json.JSONDecodeError):
            json.loads(path.read_text())

    def test_non_object_line_is_a_bad_line(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text('[1, 2, 3]\n')
        items = list(ledger.read(path))
        assert len(items) == 1 and isinstance(items[0], ledger.BadLine)
        assert "not an object" in items[0].error

    def test_blank_lines_are_skipped_silently(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger.append(good_row(), path)
        with open(path, "a") as fh:
            fh.write("\n\n")
        assert len(ledger.rows(path)) == 1

    def test_missing_file_yields_nothing(self, tmp_path):
        assert ledger.rows(tmp_path / "nope.jsonl") == []


class TestAudit:
    def test_counts_rows_not_files(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger.append(good_row(arm="a"), path)
        with open(path, "a") as fh:
            fh.write("garbage\n")
            # A row that bypassed append() and is missing provenance.
            fh.write(json.dumps({"arm": "sneaky", "accuracy_over_all": 1.0}) + "\n")

        result = ledger.audit(path)
        assert result["rows_read"] == 2
        assert len(result["bad_lines"]) == 1
        assert len(result["rows_missing_provenance"]) == 1
        assert not result["clean"]

    def test_clean_ledger_reports_clean(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        ledger.append(good_row(), path)
        assert ledger.audit(path)["clean"]

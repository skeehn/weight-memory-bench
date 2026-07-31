"""BEAM loader tests.

These exercise the parts that must be right *before* anything is downloaded: the download
gate, the preflight ordering, and the defensive probe parsing. The loader's behaviour
against real rows is unverified until a first fetch, which is deliberate -- the point of the
gate is that the fetch is a decision, not a side effect of an import.
"""

import json

import pytest

from data import beam


class TestDownloadGate:
    def test_load_refuses_without_opt_in(self, tmp_path, monkeypatch):
        monkeypatch.setattr(beam, "CACHE_DIR", tmp_path / "empty")
        monkeypatch.delenv("WMB_ALLOW_DOWNLOAD", raising=False)
        pytest.importorskip("datasets")
        with pytest.raises(beam.DownloadNotPermitted):
            beam.load("500K")

    def test_env_var_is_an_accepted_opt_in(self, monkeypatch):
        monkeypatch.setenv("WMB_ALLOW_DOWNLOAD", "1")
        assert beam._download_allowed(False)

    def test_explicit_flag_is_an_accepted_opt_in(self, monkeypatch):
        monkeypatch.delenv("WMB_ALLOW_DOWNLOAD", raising=False)
        assert beam._download_allowed(True)
        assert not beam._download_allowed(False)

    def test_unknown_split_raises_before_anything_else(self):
        # Fails on the split name without needing `datasets`, the cache, or the network.
        with pytest.raises(ValueError) as exc:
            beam.load("128K")
        assert "128K" in str(exc.value)


class TestPreflight:
    def test_passes_when_space_is_available(self, tmp_path):
        result = beam.preflight(tmp_path, required=1)
        assert result["ok"] and result["free_bytes"] > 0

    def test_raises_when_space_is_short(self, tmp_path):
        with pytest.raises(beam.InsufficientDiskSpace):
            beam.preflight(tmp_path, required=10**18)

    def test_works_on_a_path_that_does_not_exist_yet(self, tmp_path):
        # The cache dir is usually absent on a first run; the check walks up to a real
        # ancestor rather than crashing or silently skipping.
        missing = tmp_path / "a" / "b" / "c"
        result = beam.preflight(missing, required=1)
        assert result["checked"] == str(tmp_path)


class TestProbeParsing:
    def test_parses_a_json_list_payload(self):
        payload = json.dumps(
            [
                {"question": "When did they move?", "answer": "March", "ability": "temporal reasoning"},
                {"question": "What is unknowable?", "answer": "", "ability": "abstention"},
            ]
        )
        probes = beam._parse_probes("c1", payload)
        assert len(probes) == 2
        assert probes[0].ability == "temporal reasoning"
        assert probes[1].is_abstention

    def test_accepts_alternate_field_names(self):
        payload = json.dumps([{"probing_question": "q?", "gold_answer": "a", "type": "recall"}])
        probes = beam._parse_probes("c1", payload)
        assert len(probes) == 1 and probes[0].question == "q?" and probes[0].answer == "a"

    def test_unparseable_payload_yields_no_probes_rather_than_a_guess(self):
        # A fabricated probe would enter the denominator of every arm's score.
        assert beam._parse_probes("c1", "not json at all") == ()
        assert beam._parse_probes("c1", None) == ()

    def test_items_without_a_question_are_dropped(self):
        payload = json.dumps([{"answer": "orphan"}, {"question": "real?", "answer": "yes"}])
        probes = beam._parse_probes("c1", payload)
        assert len(probes) == 1 and probes[0].question == "real?"

    def test_missing_ability_defaults_to_unknown_not_abstention(self):
        # Defaulting to abstention would silently inflate the abstention denominator.
        probes = beam._parse_probes("c1", json.dumps([{"question": "q", "answer": "a"}]))
        assert probes[0].ability == "unknown" and not probes[0].is_abstention


class TestStats:
    def test_counts_probes_across_conversations(self):
        def conv(cid, abilities):
            return beam.Conversation(
                conversation_id=cid,
                category="Coding",
                chat=("turn one", "turn two"),
                probes=tuple(
                    beam.Probe(conversation_id=cid, ability=a, question="q", answer="a")
                    for a in abilities
                ),
            )

        convs = [conv("c1", ["abstention", "recall"]), conv("c2", ["recall"])]
        s = beam.stats(convs)
        assert s == {
            "conversations": 2,
            "probes": 3,
            "abstention_probes": 1,
            "by_ability": {"abstention": 1, "recall": 2},
        }

    def test_transcript_joins_every_turn(self):
        c = beam.Conversation("c1", "Math", ("alpha", "beta"), ())
        assert "alpha" in c.transcript_text() and "beta" in c.transcript_text()

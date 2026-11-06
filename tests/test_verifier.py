"""The trim-context verdict: a gold run holds every constraint, and each one fails on its own
case with its own message; plus the per-call checks.

The cases are the five a verifier has to decide -- correct, wrong, missing, malformed, and the
boundaries where the policy parts company with a reasonable guess.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from helpers import CONVERSATIONS, Run, failed, stage
from slimconfig import build_output, plain_output

import verifier
from contract import ConversationsOutput, KeyOutput, ReportInput
from plugin import drop_oldest
from verifier.task import TEXT

CONSTRAINTS = len(TEXT)


# ----- Correct -----


@pytest.mark.parametrize("name", ["conversations.json", "conversations_edges.json"])
def test_a_gold_run_holds_every_constraint(tmp_path: Path, name: str) -> None:
    run = Run(tmp_path, conversations=name)
    run.gold()
    found = run.verdict()
    assert found["valid"], found["message"]
    assert found["score"] == 1.0 and found["message"].startswith(f"All {CONSTRAINTS} checks hold")
    assert [entry["ok"] for entry in found["details"]["constraints"]] == [True] * CONSTRAINTS
    assert len(json.dumps(found)) < 8000, "the verdict lands in the chat's context: keep it small"


def test_the_file_holds_one_of_each_status(tmp_path: Path) -> None:
    run = Run(tmp_path)
    assert {entry.status.value for entry in run.key.trims} == {"unchanged", "trimmed", "overflow"}


# ----- Wrong -----


def test_a_wrong_eviction_order_fails_evicted_exact(tmp_path: Path) -> None:
    """The evicted list is the order things went, not the conversation's order."""
    run = Run(tmp_path)
    run.read()
    trims = run.answer()
    two_phase = next(entry for entry in trims if entry["conversation"] == "two_phase")
    two_phase["evicted"] = sorted(two_phase["evicted"])
    run.report(trims)
    found = run.verdict()
    assert failed(found) == ["evicted-exact"]
    assert "two_phase: the evicted list reads" in found["message"]
    assert found["score"] == pytest.approx(3 / 4)


def test_a_wrong_status_fails_status_exact(tmp_path: Path) -> None:
    run = Run(tmp_path)
    run.read()
    trims = run.answer()
    next(entry for entry in trims if entry["conversation"] == "overflow")["status"] = "trimmed"
    run.report(trims)
    found = run.verdict()
    assert "status-exact" in failed(found) and "overflow: the status reads" in found["message"]


def test_a_right_eviction_with_a_wrong_total_fails_only_total_exact(tmp_path: Path) -> None:
    run = Run(tmp_path)
    run.read()
    trims = run.answer()
    next(entry for entry in trims if entry["conversation"] == "one_over")["total"] += 5
    run.report(trims)
    found = run.verdict()
    assert failed(found) == ["total-exact"], failed(found)


def test_trusting_drop_oldest_throws_the_system_prompt_away(tmp_path: Path) -> None:
    """drop_oldest is honest about dropping the oldest and knows nothing of protection."""
    run = Run(tmp_path)
    dropped = drop_oldest(run.given, tmp_path)
    over = next(entry for entry in dropped.results if entry.conversation == "two_phase")
    assert "s1" not in over.kept, "the oldest message is the system prompt, and it goes first"
    run.read()
    trims = run.answer()
    target = next(entry for entry in trims if entry["conversation"] == "two_phase")
    target["kept"] = list(over.kept)
    target["evicted"] = [i for i in (m for m in ("s1", "u1", "a1", "c1", "r1", "a2", "c2", "r2", "a3", "u2")) if i not in over.kept]
    target["total"] = over.total
    run.report(trims)
    found = run.verdict()
    assert "kept-exact" in failed(found) and found["score"] < 1


def test_a_submission_that_is_not_the_saved_report_fails_submitted_as_saved(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    changed = copy.deepcopy(report)
    changed["trims"][0]["total"] += 1
    found = run.verdict(changed)
    assert failed(found)[0] == "submitted-as-saved" and found["score"] == 0.0


def test_an_answer_key_that_breaks_the_policy_fails_answer_key_right(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    key = plain_output(run.key)
    key["trims"][0] = {**key["trims"][0], "status": "overflow"}
    found = run.verdict(report, reference=build_output(key, KeyOutput))
    assert failed(found)[0] == "answer-key-right" and found["score"] == 0.0


# ----- Missing -----


def test_no_answer_key_fails_first_and_scores_zero(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    found = run.verdict(report, reference=None)
    assert failed(found)[0] == "answer-key" and found["score"] == 0.0 and "trim_key never read" in found["message"]


def test_trimming_without_reading_the_file_fails_read_conversations(tmp_path: Path) -> None:
    run = Run(tmp_path)
    run.gold()
    (tmp_path / "reports" / run.run / "conversations.json").unlink()
    found = run.verdict()
    assert failed(found)[0] == "read-conversations" and found["score"] == 0.0
    assert "was never called in run" in found["message"]


def test_an_unreported_conversation_fails_all_conversations(tmp_path: Path) -> None:
    run = Run(tmp_path)
    run.read()
    run.report(run.answer()[:-1])
    found = run.verdict()
    assert failed(found) == ["all-conversations"]
    assert "1 of the 4 conversations is not reported (overflow)" in found["message"]
    assert found["score"] == pytest.approx(3 / 4)


def test_no_saved_report_fails_report_saved(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    (tmp_path / "reports" / run.run / "trims.json").unlink()
    found = run.verdict(report)
    assert failed(found)[0] == "report-saved" and found["score"] == 0.0


def test_a_missing_source_file_fails_sources(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    (tmp_path / CONVERSATIONS).unlink()
    found = run.verdict(report)
    assert "sources" in failed(found) and "No file at conversations.json" in found["message"]


# ----- Malformed -----


def test_an_orphan_tool_result_fails_sources(tmp_path: Path) -> None:
    """Rule 5: the file is refused rather than producing some other status."""
    run = Run(tmp_path)
    report = run.gold()
    stage(tmp_path, "conversations_orphan.json")
    broken = json.loads((tmp_path / "conversations_orphan.json").read_text())
    (tmp_path / CONVERSATIONS).write_text(json.dumps(broken))
    found = run.verdict(report)
    assert "sources" in failed(found)
    assert "a tool_result answers a tool_call that comes before it" in found["message"]


def test_a_repeated_id_fails_sources(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    path = tmp_path / CONVERSATIONS
    data = json.loads(path.read_text())
    data["conversations"][0]["messages"][1]["id"] = data["conversations"][0]["messages"][0]["id"]
    path.write_text(json.dumps(data))
    found = run.verdict(report)
    assert "sources" in failed(found) and "is used twice" in found["message"]


def test_a_message_both_kept_and_evicted_fails_submitted_shape(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    broken = copy.deepcopy(report)
    entry = next(e for e in broken["trims"] if e["evicted"])
    entry["kept"].append(entry["evicted"][0])
    found = run.verdict(broken)
    assert failed(found) == ["submitted-shape"] and "both kept and evicted" in found["message"]


def test_an_id_that_is_not_in_the_conversation_fails_submitted_shape(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    broken = copy.deepcopy(report)
    broken["trims"][0]["kept"].append("nope")
    found = run.verdict(broken)
    assert failed(found) == ["submitted-shape"] and "'nope' is not a message in it" in found["message"]


def test_a_message_neither_kept_nor_evicted_fails_submitted_shape(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    broken = copy.deepcopy(report)
    broken["trims"][0]["kept"] = broken["trims"][0]["kept"][1:]
    found = run.verdict(broken)
    assert failed(found) == ["submitted-shape"] and "either kept or evicted" in found["message"]


def test_a_conversation_reported_twice_fails_submitted_shape(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    twice = copy.deepcopy(report)
    twice["trims"].append(copy.deepcopy(twice["trims"][0]))
    found = run.verdict(twice)
    assert failed(found) == ["submitted-shape"] and "is reported twice" in found["message"]


@pytest.mark.parametrize(("token", "words"), [("../..", "is not a run token"), ("", "is not a run token"), ("0" * 16, "has no run")])
def test_a_run_that_is_not_this_workspace_s_fails_this_run(tmp_path: Path, token: str, words: str) -> None:
    run = Run(tmp_path)
    report = run.gold()
    found = run.verdict({**report, "run": token})
    assert failed(found)[0] == "this-run" and words in found["message"] and found["score"] == 0.0


def test_an_edited_report_file_always_gives_a_verdict(tmp_path: Path) -> None:
    """600 mutations of the saved report: one field, at any depth, replaced by an odd JSON value
    or dropped. Every one is a verdict; none is an exception."""
    import random

    run = Run(tmp_path)
    report = run.gold()
    path = tmp_path / "reports" / run.run / "trims.json"
    good = json.loads(path.read_text())
    odd = [None, 0, -1, 1e308, "", "x", [], {}, [1], {"a": 1}, True, "../..", "s1", ["s1", 2]]
    randomly = random.Random(17)
    for _ in range(600):
        broken = json.loads(json.dumps(good))
        target = randomly.choice(["task", "trims", "entry", "field", "drop"])
        if target in ("entry", "field") and broken["trims"]:
            key = randomly.choice(list(broken["trims"]))
            if target == "entry":
                broken["trims"][key] = randomly.choice(odd)
            else:
                field = randomly.choice(["kept", "evicted", "total", "status"])
                broken["trims"][key][field] = randomly.choice(odd)
        elif target == "drop":
            broken.pop(randomly.choice(list(broken)), None)
        else:
            broken[target] = randomly.choice(odd)
        path.write_text(json.dumps(broken))
        found = run.verdict(report)
        assert isinstance(found["valid"], bool) and 0.0 <= found["score"] <= 1.0, broken


# ----- Boundary -----


def test_exactly_at_the_limit_is_unchanged(tmp_path: Path) -> None:
    run = Run(tmp_path)
    steady = next(entry for entry in run.key.trims if entry.conversation == "steady")
    assert steady.status.value == "unchanged" and steady.evicted == [] and steady.total == 120


def test_one_token_over_evicts_exactly_one(tmp_path: Path) -> None:
    run = Run(tmp_path)
    one_over = next(entry for entry in run.key.trims if entry.conversation == "one_over")
    assert one_over.evicted == ["r1"] and one_over.status.value == "trimmed"
    assert "c1" in one_over.kept, "the tool_call its result answered stays"


def test_the_second_pass_begins_when_the_first_is_not_enough(tmp_path: Path) -> None:
    run = Run(tmp_path)
    two_phase = next(entry for entry in run.key.trims if entry.conversation == "two_phase")
    assert two_phase.evicted[:2] == ["r1", "r2"], "both tool_results first, oldest first"
    assert two_phase.evicted[2:] == ["u1", "a1", "c1", "a2"], "then the oldest unprotected, stopping when it fits"
    assert two_phase.total == 83 and two_phase.status.value == "trimmed"


def test_protected_messages_alone_over_the_limit_is_overflow(tmp_path: Path) -> None:
    run = Run(tmp_path)
    over = next(entry for entry in run.key.trims if entry.conversation == "overflow")
    assert over.status.value == "overflow"
    assert over.kept == ["s1", "u2"] and over.total == 85, "only the protected are left"


def test_a_huge_last_user_message_is_still_protected(tmp_path: Path) -> None:
    run = Run(tmp_path, conversations="conversations_edges.json")
    huge = next(entry for entry in run.key.trims if entry.conversation == "huge_user")
    assert huge.status.value == "overflow" and "u1" in huge.kept and huge.total == 210
    assert run.gold() and run.verdict()["valid"]


def test_a_conversation_with_no_system_message(tmp_path: Path) -> None:
    run = Run(tmp_path, conversations="conversations_edges.json")
    none = next(entry for entry in run.key.trims if entry.conversation == "no_system")
    assert none.kept == ["u2"], "only the last user message is protected here"


def test_a_pending_tool_call_is_evicted_on_its_own(tmp_path: Path) -> None:
    run = Run(tmp_path, conversations="conversations_edges.json")
    pending = next(entry for entry in run.key.trims if entry.conversation == "call_without_result")
    assert pending.evicted == ["c1", "a1"], "nothing to take with it, so the call goes alone"


# ----- The per-call checks -----


def test_every_tool_has_both_checks_on_its_module() -> None:
    from contract import Tool

    assert set(verifier.TOOLS) == set(Tool)
    for tool in Tool:
        module = verifier.TOOLS[tool]
        for name in ("verify_input", "verify_output"):
            assert callable(module.__dict__.get(name)), f"{tool.value}: {module.__name__} has no {name}"


def test_verify_input_answers_for_every_tool(tmp_path: Path) -> None:
    run = Run(tmp_path)
    from contract import Tool

    valid = {
        Tool.READ_CONVERSATIONS: run.given,
        Tool.TRIM_KEY: run.given,
        Tool.DROP_OLDEST: run.given,
        Tool.REPORT_TRIMS: build_output({"task": run.task, "trims": []}, ReportInput),
    }
    for tool, value in valid.items():
        found = verifier.verify_input(tool.value, value)
        assert found["valid"] is True, (tool.value, found["message"])


def test_verify_output_recomputes_the_conversations(tmp_path: Path) -> None:
    run = Run(tmp_path)
    data = plain_output(run.read())
    assert verifier.verify_output("read_conversations", run.given, build_output(data, ConversationsOutput), str(tmp_path))["valid"]
    changed = copy.deepcopy(data)
    changed["conversations"][0]["limit"] += 1
    found = verifier.verify_output("read_conversations", run.given, build_output(changed, ConversationsOutput), str(tmp_path))
    assert not found["valid"] and "the limit is not the file's" in found["message"]


def test_the_distractor_is_checked_against_its_own_rule(tmp_path: Path) -> None:
    run = Run(tmp_path)
    found = drop_oldest(run.given, tmp_path)
    assert verifier.verify_output("drop_oldest", run.given, found, str(tmp_path))["valid"]


# ----- The verdict's own shape -----


def test_every_verdict_entry_has_the_same_keys_and_says_why(tmp_path: Path) -> None:
    run = Run(tmp_path)
    report = run.gold()
    broken = copy.deepcopy(report)
    broken["trims"][1]["status"] = "unchanged"
    for found in (run.verdict(report), run.verdict(broken), run.verdict({**report, "run": "0" * 16})):
        entries = found["details"]["constraints"]
        order = [entry["id"] for entry in entries]
        assert order == list(TEXT)
        for position, entry in enumerate(entries):
            assert set(entry) == {"id", "constraint", "ok", "expected", "actual", "message"}
            assert (entry["message"] == "") == (entry["ok"] is True)
            if entry["ok"] is None:
                blocking = entry["message"].removeprefix("Not checked: ").removesuffix(" did not hold.").split(" and ")
                assert all(name in order[:position] for name in blocking), entry
        wrong = [entry for entry in entries if entry["ok"] is False]
        if wrong:
            more = f" (+{len(wrong) - 1} more)" if len(wrong) > 1 else ""
            assert found["message"] == f"{wrong[0]['id']}: {wrong[0]['message'].removesuffix('.')}{more}"


def test_the_readme_lists_every_constraint_as_the_verifier_names_it() -> None:
    import re

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\d+) \| `([a-z-]+)` \| ([^|]+) \|", readme, re.MULTILINE)
    assert [(int(n), name, text.strip()) for n, name, text in rows] == [(n, *e) for n, e in enumerate(TEXT.items(), 1)]

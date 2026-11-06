"""The trim-context verdict: 14 constraints, in order, on the answer key, the run the submission
names, the receipt read_conversations left, the saved and submitted report, and the trimming.

A failure among the first 9 scores 0; otherwise the score is the share of the conversations
whose every value is right. The kept list, the evicted list and the total are checked apart,
because they fail for different reasons: the kept list is the conversation's own order, the
evicted list is the order things went, and the total is arithmetic over the kept ids.
"""

from pathlib import Path

from .checks import RECEIPT, REPORT, constraint, read_json, result, run_folder, run_problem, summary
from .rules import expected, load

TEXT = {
    "answer-key": "The answer key is for this task's file",
    "sources": "The conversations file is well formed",
    "answer-key-right": "The answer key follows the eviction policy",
    "this-run": "The report names a run of this workspace",
    "read-conversations": "read_conversations read this task's file in this run",
    "report-saved": "report_trims saved this run's report for this task",
    "submitted-task": "The submitted report is for this task",
    "submitted-shape": "The submitted report is well formed",
    "submitted-as-saved": "The submitted report equals the saved report",
    "all-conversations": "Every conversation is reported exactly once, in the file's order",
    "status-exact": "Each status is right",
    "kept-exact": "Each kept list is right, in the conversation's own order",
    "evicted-exact": "Each evicted list is right, in the order things were evicted",
    "total-exact": "Each total is what the kept messages add up to",
}
ORDER = tuple(TEXT)
GATES = ORDER[:9]
ANSWER_CHECKS = ORDER[9:]
MOST_NAMED = 3
SUBMIT = "Submit the report the last report_trims call returned, as it is."


def named(names):
    shown = ", ".join(str(name) for name in names[:MOST_NAMED])
    return shown + (f" and {len(names) - MOST_NAMED} more" if len(names) > MOST_NAMED else "")


def also(words, names):
    return f" Also {words}: {named(names[1:])}." if len(names) > 1 else ""


def saved_report(workspace, run):
    where = f"{run_folder(run)}/{REPORT}"
    found, problem = read_json(Path(workspace) / where)
    if problem == "missing":
        return None, f"no report at {where}: report_trims never saved this task's trimming"
    if problem:
        return None, f"{where} is not a report: {problem}"
    if not isinstance(found, dict) or not isinstance(found.get("trims"), dict) or not isinstance(found.get("task"), dict):
        return None, f"{where} was altered after report_trims: it must hold task and trims"
    for name, entry in found["trims"].items():
        if not isinstance(entry, dict):
            return None, f"{where} was altered after report_trims: {name}'s entry must hold its lists"
        for field in ("kept", "evicted"):
            if not isinstance(entry.get(field), list) or any(not isinstance(v, str) for v in entry[field]):
                return None, f"{where} was altered after report_trims: {name}'s {field} must be message ids"
        if not isinstance(entry.get("total"), int) or isinstance(entry.get("total"), bool):
            return None, f"{where} was altered after report_trims: {name}'s total must be a whole number"
        if not isinstance(entry.get("status"), str):
            return None, f"{where} was altered after report_trims: {name}'s status must be a name"
    return found, None


def given_trims(output):
    return [
        {
            "conversation": entry.conversation,
            "kept": list(entry.kept),
            "evicted": list(entry.evicted),
            "total": entry.total,
            "status": entry.status.value,
        }
        for entry in output.trims
    ]


def shape_problem(conversations, trims):
    """Why the submitted report cannot be judged, or None."""
    known = {conversation["name"]: {message["id"] for message in conversation["messages"]} for conversation in conversations}
    listed = [entry["conversation"] for entry in trims]
    unknown = [name for name in listed if name not in known]
    if unknown:
        return f"{unknown[0]!r} is not a conversation in this file"
    repeated = [name for name in listed if listed.count(name) > 1]
    if repeated:
        return f"{repeated[0]} is reported twice"
    for entry in trims:
        ids = known[entry["conversation"]]
        for field in ("kept", "evicted"):
            outside = [value for value in entry[field] if value not in ids]
            if outside:
                return f"{entry['conversation']}: {outside[0]!r} is not a message in it"
            if len(set(entry[field])) != len(entry[field]):
                return f"{entry['conversation']}: {field} names a message twice"
        both = set(entry["kept"]) & set(entry["evicted"])
        if both:
            return f"{entry['conversation']}: {min(both)} is both kept and evicted"
        if set(entry["kept"]) | set(entry["evicted"]) != ids:
            return f"{entry['conversation']}: every message is either kept or evicted"
        if entry["total"] < 0:
            return f"{entry['conversation']}: the total is negative"
    return None


def verify_task(arguments, output, reference, workspace):
    """The verdict on a submitted trim-context report (see the module's docstring)."""
    checks = {}

    def put(identifier, ok, expected_value=None, actual=None, message=""):
        checks[identifier] = constraint(identifier, TEXT[identifier], ok, expected_value, actual, message)

    def skip(identifier, *needs):
        blocking = [need for need in needs if checks[need]["ok"] is not True]
        checks[identifier] = constraint(identifier, TEXT[identifier], None, message=f"Not checked: {' and '.join(blocking)} did not hold.")

    def hold(*identifiers):
        return all(checks[identifier]["ok"] is True for identifier in identifiers)

    # 1-3: the source and the answer key.
    conversations, source, problem = load(workspace, arguments.conversations)
    if reference is None:
        put("answer-key", False, None, None, f"No answer key: trim_key never read {arguments.conversations}.")
    elif problem is None and reference.source_sha256 != source:
        put("answer-key", False, source[:16], reference.source_sha256[:16], "The answer key read another file than this task names.")
    else:
        put("answer-key", True)
    put("sources", problem is None, None, None, problem or "")
    wanted, names = [], []
    if hold("answer-key", "sources"):
        wanted = expected(conversations)
        names = [entry["conversation"] for entry in wanted]
        put("answer-key-right", given_trims(reference) == wanted, None, None, "The answer key is not what the policy gives.")
    else:
        skip("answer-key-right", "answer-key", "sources")

    # 4-5: the run, and the receipt read_conversations left in it.
    run = output.run
    problem = run_problem(workspace, run)
    put("this-run", problem is None, None, run, problem or "")
    if hold("this-run", "sources"):
        where = f"{run_folder(run)}/{RECEIPT}"
        found, trouble = read_json(Path(workspace) / where)
        if trouble == "missing":
            message = f"read_conversations was never called in run {run} (no {where}): read the file before trimming."
        elif trouble:
            message = f"{where} is not a receipt: {trouble}."
        elif not isinstance(found, dict) or found.get("run") != run:
            message = f"{where} was altered after read_conversations: it does not name run {run}."
        elif found.get("conversations") != arguments.conversations:
            message = f"read_conversations read {found.get('conversations')!r}, not this task's {arguments.conversations!r}."
        else:
            message = ""
        put("read-conversations", not message, None, None, message)
    else:
        skip("read-conversations", "this-run", "sources")

    # 6-9: the saved and the submitted report.
    saved = None
    if hold("this-run"):
        saved, problem = saved_report(workspace, run)
        wanted_task = {"conversations": arguments.conversations}
        if saved is not None and saved["task"] != wanted_task:
            saved, problem = None, f"the saved report is for another task: {saved['task']}"
        put("report-saved", saved is not None, None, None, problem or "")
    else:
        skip("report-saved", "this-run")
    given_task = {"conversations": output.task.conversations}
    wanted_task = {"conversations": arguments.conversations}
    put("submitted-task", given_task == wanted_task, wanted_task, given_task, f"The submitted report names {given_task}, not this task's. {SUBMIT}")
    trims = given_trims(output)
    if hold("sources", "submitted-task"):
        problem = shape_problem(conversations, trims)
        put("submitted-shape", problem is None, None, None, f"The submitted report is malformed: {problem}" if problem else "")
    else:
        skip("submitted-shape", "sources", "submitted-task")
    if saved is not None and hold("submitted-shape"):
        held = saved["trims"]
        difference = ""
        for entry in trims:
            name = entry["conversation"]
            if name not in held:
                difference = f"it adds {name}, which the saved report does not hold"
            elif {key: held[name].get(key) for key in entry if key != "conversation"} != {k: v for k, v in entry.items() if k != "conversation"}:
                difference = f"{name} reads differently in the submission than in the saved report"
            if difference:
                break
        if not difference and len(held) != len(trims):
            difference = "it leaves out a conversation the saved report holds"
        put("submitted-as-saved", not difference, len(held), len(trims), f"The submitted report is not the saved report {run_folder(run)}/{REPORT}: {difference}. {SUBMIT}")
    else:
        skip("submitted-as-saved", "report-saved", "submitted-shape")

    # 10-14: the trimming itself.
    right = None
    if hold("answer-key-right", "submitted-shape"):
        by_name = {entry["conversation"]: entry for entry in trims}
        listed = [entry["conversation"] for entry in trims]
        missing = [name for name in names if name not in by_name]
        if missing or listed != [name for name in names if name in by_name]:
            message = (
                f"{len(missing)} of the {len(names)} conversations {'is' if len(missing) == 1 else 'are'} not reported ({named(missing)})."
                if missing
                else "the conversations are not in the file's order"
            )
            put("all-conversations", False, len(names), len(by_name), message)
        else:
            put("all-conversations", True)
        for identifier, field, describe in (
            ("status-exact", "status", "status"),
            ("kept-exact", "kept", "kept list"),
            ("evicted-exact", "evicted", "evicted list"),
            ("total-exact", "total", "total"),
        ):
            bad = [want for want in wanted if want["conversation"] in by_name and by_name[want["conversation"]][field] != want[field]]
            if bad:
                first = bad[0]
                got = by_name[first["conversation"]][field]
                put(
                    identifier,
                    False,
                    {first["conversation"]: first[field]},
                    {first["conversation"]: got},
                    f"{first['conversation']}: the {describe} reads {got}, but the policy gives {first[field]}.{also('wrong', [entry['conversation'] for entry in bad])}",
                )
            else:
                put(identifier, True)
        right = sum(1 for want in wanted if by_name.get(want["conversation"]) == want)
    else:
        for identifier in ANSWER_CHECKS:
            skip(identifier, "answer-key-right", "submitted-shape")

    constraints = [checks[identifier] for identifier in ORDER]
    valid = all(entry["ok"] is True for entry in constraints)
    if any(checks[identifier]["ok"] is not True for identifier in GATES) or right is None or not names:
        score = 0.0
    else:
        score = right / len(names)
    details = {"constraints": constraints, "conversations": len(names) or None}
    if right is not None:
        details["right"] = right
    if valid:
        kinds = {}
        for want in wanted:
            kinds[want["status"]] = kinds.get(want["status"], 0) + 1
        shown = ", ".join(f"{count} {status}" for status, count in sorted(kinds.items()))
        return result(True, f"All {len(constraints)} checks hold: {len(names)} conversations trimmed by the policy ({shown}).", score=1.0, **details)
    return result(False, summary(constraints), score=min(score, 0.99), **details)

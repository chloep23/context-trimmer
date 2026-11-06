"""report_trims: merge this run's per-conversation results into its report file."""

from __future__ import annotations

import json
from pathlib import Path

from contract import ReportInput, ReportOutput, Status, Trim, TrimsOutput
from plugin.rules import RuleError, load_sources
from plugin.trimming import REPORTS, RUN, run_folder

REPORT = "trims.json"


def report_path(run: str = RUN) -> str:
    return f"{REPORTS}/{run}/{REPORT}"


def report_trims(given: ReportInput, workspace: Path) -> ReportOutput:
    conversations, _ = load_sources(Path(workspace), given.task.conversations)
    order = [conversation.name for conversation in conversations]
    named = [entry.conversation for entry in given.trims]
    unknown = [name for name in named if name not in set(order)]
    if unknown:
        raise RuleError(f"the file has no conversation {unknown[0]!r}")
    if len(set(named)) != len(named):
        raise RuleError("a conversation is reported twice in one call")

    folder = run_folder(Path(workspace))
    path = folder / REPORT
    held = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"trims": {}}
    wanted = {"conversations": given.task.conversations}
    if held.get("task") not in (None, wanted):
        raise RuleError(f"{report_path()} is another task's report: start a new run")
    for entry in given.trims:
        held["trims"][entry.conversation] = {
            "kept": list(entry.kept),
            "evicted": list(entry.evicted),
            "total": entry.total,
            "status": entry.status.value,
        }
    held["task"] = wanted
    path.write_text(json.dumps(held, indent=1), encoding="utf-8")

    trims = [
        Trim(conversation=name, kept=held["trims"][name]["kept"], evicted=held["trims"][name]["evicted"],
             total=held["trims"][name]["total"], status=Status(held["trims"][name]["status"]))
        for name in order
        if name in held["trims"]
    ]
    report = TrimsOutput(task=given.task, run=RUN, trims=trims)
    return ReportOutput(path=report_path(), covered=len(trims), complete=len(trims) == len(order), report=report)

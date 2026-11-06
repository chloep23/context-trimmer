"""Staging a workspace and running the prescribed path, as the host runs a trim-context task."""

from __future__ import annotations

import shutil
from pathlib import Path

from slimconfig import build_output, plain_output

import verifier
from contract import KeyOutput, ReportInput, TrimInput, TrimsOutput
from plugin import read_conversations, report_trims, trim_key

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONVERSATIONS = "conversations.json"


def stage(workspace: Path, *names: str) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy(FIXTURES / name, workspace / name)


class Run:
    """One task's workspace: the staged file, the tools, and the verdict on what was submitted."""

    def __init__(self, workspace: Path, conversations: str = CONVERSATIONS) -> None:
        self.workspace = workspace
        self.task = {"conversations": conversations}
        stage(workspace, conversations)
        self.given = build_output(self.task, TrimInput)
        self.submitted: dict | None = None
        self.run = ""
        self.key: KeyOutput = build_output(plain_output(trim_key(self.given, workspace)), KeyOutput)

    def read(self):
        found = read_conversations(self.given, self.workspace)
        self.run = found.run
        return found

    def report(self, trims: list[dict]) -> dict:
        given = build_output({"task": self.task, "trims": trims}, ReportInput)
        self.submitted = plain_output(report_trims(given, self.workspace).report)
        return self.submitted

    def answer(self) -> list[dict]:
        """The answer key, in the shape report_trims takes."""
        return [
            {
                "conversation": entry.conversation,
                "kept": list(entry.kept),
                "evicted": list(entry.evicted),
                "total": entry.total,
                "status": entry.status.value,
            }
            for entry in self.key.trims
        ]

    def gold(self) -> dict:
        self.read()
        return self.report(self.answer())

    def verdict(self, report: dict | None = None, reference: KeyOutput | None = ...) -> dict:
        found = report if report is not None else self.submitted
        return verifier.verify_task(
            "trim-context",
            self.given,
            build_output(found, TrimsOutput),
            self.key if reference is ... else reference,
            str(self.workspace),
        )


def failed(verdict: dict) -> list[str]:
    return [entry["id"] for entry in verdict["details"]["constraints"] if entry["ok"] is False]


def unchecked(verdict: dict) -> list[str]:
    return [entry["id"] for entry in verdict["details"]["constraints"] if entry["ok"] is None]

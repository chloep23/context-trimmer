"""The tools: read_conversations, trim_key (the reference) and drop_oldest (a cruder policy)."""

from __future__ import annotations

import json
import secrets
from pathlib import Path

from contract import Conversation, ConversationsOutput, DropOldestOutput, Dropped, KeyOutput, Trim, TrimInput
from plugin.rules import drop_oldest as drop
from plugin.rules import load_sources, trim

RUN = secrets.token_hex(8)
REPORTS = "reports"
MARKER = "run.json"
RECEIPT = "conversations.json"


def run_folder(workspace: Path, run: str = RUN) -> Path:
    folder = Path(workspace) / REPORTS / run
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / MARKER
    if not marker.is_file():
        marker.write_text(json.dumps({"run": run}), encoding="utf-8")
    return folder


def read_conversations(given: TrimInput, workspace: Path) -> ConversationsOutput:
    """Every conversation and its limit. It evicts nothing of its own."""
    conversations, source = load_sources(Path(workspace), given.conversations)
    receipt = {"run": RUN, "conversations": given.conversations, "sha256": source}
    (run_folder(Path(workspace)) / RECEIPT).write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return ConversationsOutput(run=RUN, conversations=conversations, source_sha256=source)


def as_trim(conversation: Conversation) -> Trim:
    made = trim(conversation)
    return Trim(
        conversation=conversation.name,
        kept=made.kept,
        evicted=made.evicted,
        total=made.total,
        status=made.status,
    )


def trim_key(given: TrimInput, workspace: Path) -> KeyOutput:
    """The policy applied to every conversation. The task's reference: the agent never sees it."""
    conversations, source = load_sources(Path(workspace), given.conversations)
    return KeyOutput(source_sha256=source, trims=[as_trim(conversation) for conversation in conversations])


def drop_oldest(given: TrimInput, workspace: Path) -> DropOldestOutput:
    """The oldest messages go until the limit is met, whatever their role.

    It knows nothing of what a message is for: the oldest message is almost always the system
    prompt, which the policy protects, so what it leaves is not what the policy leaves.
    """
    conversations, _ = load_sources(Path(workspace), given.conversations)
    results = []
    for conversation in conversations:
        kept, total = drop(conversation)
        results.append(Dropped(conversation=conversation.name, kept=kept, total=total))
    return DropOldestOutput(results=results)

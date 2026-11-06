"""The eviction policy, applied one step at a time.

Protected messages are never evicted: every system message, and the last user message. If the
conversation already fits the limit nothing moves. Otherwise items go one at a time, and the
total is checked after each one, so no more is evicted than the limit needs: first the
tool_result messages oldest first, their tool_calls staying put, and then, if it is still over,
the remaining unprotected messages oldest first.

Evicting a tool_call takes its tool_result with it, as one step, when that result is still
there. In practice it never is: the first phase evicts every tool_result, and the second phase
only runs when the first was not enough. The clause is kept because the policy states it, and
``tests/test_rules.py`` pins that nothing reaches it.

When everything unprotected has gone and the total is still over the limit, only the protected
messages are left and the status is overflow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from contract import Conversation, Message, Role, Status

ROLES = {role.value for role in Role}


class RuleError(ValueError):
    """A conversations file that does not follow the format."""


@dataclass(frozen=True)
class Trimming:
    kept: list[str]
    evicted: list[str]
    total: int
    status: Status


def whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def read_conversations_data(data) -> list[Conversation]:
    """Every conversation a parsed file states, or raise ``RuleError``."""
    if not isinstance(data, dict) or not isinstance(data.get("conversations"), list) or not data["conversations"]:
        raise RuleError("a conversations file holds a conversations list with at least one conversation")
    names, made = set(), []
    for entry in data["conversations"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise RuleError("every conversation has a name")
        name = entry["name"]
        if name in names:
            raise RuleError(f"conversation {name!r} is listed twice")
        names.add(name)
        if not whole(entry.get("limit")):
            raise RuleError(f"{name}: limit is a whole number of 0 or more, not {entry.get('limit')!r}")
        if not isinstance(entry.get("messages"), list) or not entry["messages"]:
            raise RuleError(f"{name}: a conversation holds at least one message")
        ids, calls, messages = set(), set(), []
        for item in entry["messages"]:
            if not isinstance(item, dict):
                raise RuleError(f"{name}: every message is an object")
            identifier, role = item.get("id"), item.get("role")
            if not isinstance(identifier, str) or not identifier.strip():
                raise RuleError(f"{name}: every message has an id")
            if identifier in ids:
                raise RuleError(f"{name}: id {identifier!r} is used twice")
            if role not in ROLES:
                raise RuleError(f"{name}/{identifier}: role is one of {sorted(ROLES)}, not {role!r}")
            if not whole(item.get("tokens")):
                raise RuleError(f"{name}/{identifier}: tokens is a whole number of 0 or more, not {item.get('tokens')!r}")
            answers = item.get("answers")
            if role == Role.TOOL_RESULT.value:
                if not isinstance(answers, str) or answers not in calls:
                    raise RuleError(f"{name}/{identifier}: a tool_result answers a tool_call that comes before it, not {answers!r}")
            elif answers is not None:
                raise RuleError(f"{name}/{identifier}: only a tool_result answers a tool_call")
            if role == Role.TOOL_CALL.value:
                calls.add(identifier)
            ids.add(identifier)
            messages.append(Message(id=identifier, role=Role(role), tokens=item["tokens"], answers=answers))
        made.append(Conversation(name=name, limit=entry["limit"], messages=messages))
    return made


def protected_ids(messages: list[Message]) -> set[str]:
    """Every system message, and the last user message."""
    kept = {message.id for message in messages if message.role is Role.SYSTEM}
    last_user = next((message.id for message in reversed(messages) if message.role is Role.USER), None)
    if last_user is not None:
        kept.add(last_user)
    return kept


def trim(conversation: Conversation) -> Trimming:
    """The one trimming the policy gives for this conversation."""
    messages = conversation.messages
    limit = conversation.limit
    tokens = {message.id: message.tokens for message in messages}
    total = sum(tokens.values())
    order = [message.id for message in messages]
    if total <= limit:
        return Trimming(kept=list(order), evicted=[], total=total, status=Status.UNCHANGED)

    safe = protected_ids(messages)
    present = set(order)
    result_of = {message.answers: message.id for message in messages if message.role is Role.TOOL_RESULT}
    evicted: list[str] = []

    def evict(identifier: str) -> None:
        nonlocal total
        present.discard(identifier)
        evicted.append(identifier)
        total -= tokens[identifier]

    # (a) the tool_result messages, oldest first. Their tool_calls stay.
    for message in messages:
        if total <= limit:
            break
        if message.role is Role.TOOL_RESULT and message.id in present and message.id not in safe:
            evict(message.id)

    # (b) whatever unprotected is left, oldest first. A tool_call takes its result with it.
    for message in messages:
        if total <= limit:
            break
        if message.id in present and message.id not in safe:
            evict(message.id)
            answer = result_of.get(message.id)
            if message.role is Role.TOOL_CALL and answer is not None and answer in present:
                evict(answer)  # one step with the call; never reached, see the module docstring

    status = Status.OVERFLOW if total > limit else Status.TRIMMED
    return Trimming(kept=[identifier for identifier in order if identifier in present], evicted=evicted, total=total, status=status)


def drop_oldest(conversation: Conversation) -> tuple[list[str], int]:
    """What is left when the oldest messages go until the limit is met, whatever their role."""
    messages = conversation.messages
    total = sum(message.tokens for message in messages)
    kept = [message.id for message in messages]
    for message in messages:
        if total <= conversation.limit:
            break
        kept.remove(message.id)
        total -= message.tokens
    return kept, total


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inside(workspace: Path, name: str) -> Path:
    relative = Path(name)
    root = Path(workspace).resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root) or path == root:
        raise RuleError(f"{name!r} must name a file inside the workspace")
    return path


def read_text(workspace: Path, name: str) -> str:
    path = inside(workspace, name)
    if not path.is_file():
        raise RuleError(f"no file at {name}")
    return path.read_text(encoding="utf-8")


def load_sources(workspace: Path, name: str) -> tuple[list[Conversation], str]:
    text = read_text(workspace, name)
    try:
        return read_conversations_data(json.loads(text)), sha256_text(text)
    except json.JSONDecodeError as error:
        raise RuleError(f"{name} is not JSON: {error}") from error

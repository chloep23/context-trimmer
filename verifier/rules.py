"""The eviction policy, applied again from the file alone.

This is the verifier's own reading and its own policy: it shares no code with the plugin, so a
tool and the check of that tool cannot be wrong in the same way. Protected messages never move;
the total is checked after every step so nothing extra goes; tool_results go first, oldest
first, then whatever unprotected is left, oldest first.
"""

import hashlib
import json
from pathlib import Path

ROLES = ("system", "user", "assistant", "tool_call", "tool_result")
UNCHANGED, TRIMMED, OVERFLOW = "unchanged", "trimmed", "overflow"


def whole(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def parse_conversations(data):
    """(the conversations as plain values, None) or (None, what is wrong with the file)."""
    if not isinstance(data, dict) or not isinstance(data.get("conversations"), list) or not data["conversations"]:
        return None, "a conversations file holds a conversations list with at least one conversation"
    names, made = set(), []
    for entry in data["conversations"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
            return None, "every conversation has a name"
        name = entry["name"]
        if name in names:
            return None, f"conversation {name!r} is listed twice"
        names.add(name)
        if not whole(entry.get("limit")):
            return None, f"{name}: limit is a whole number of 0 or more, not {entry.get('limit')!r}"
        if not isinstance(entry.get("messages"), list) or not entry["messages"]:
            return None, f"{name}: a conversation holds at least one message"
        ids, calls, messages = set(), set(), []
        for item in entry["messages"]:
            if not isinstance(item, dict):
                return None, f"{name}: every message is an object"
            identifier, role = item.get("id"), item.get("role")
            if not isinstance(identifier, str) or not identifier.strip():
                return None, f"{name}: every message has an id"
            if identifier in ids:
                return None, f"{name}: id {identifier!r} is used twice"
            if role not in ROLES:
                return None, f"{name}/{identifier}: role is one of {sorted(ROLES)}, not {role!r}"
            if not whole(item.get("tokens")):
                return None, f"{name}/{identifier}: tokens is a whole number of 0 or more, not {item.get('tokens')!r}"
            answers = item.get("answers")
            if role == "tool_result":
                if not isinstance(answers, str) or answers not in calls:
                    return None, f"{name}/{identifier}: a tool_result answers a tool_call that comes before it, not {answers!r}"
            elif answers is not None:
                return None, f"{name}/{identifier}: only a tool_result answers a tool_call"
            if role == "tool_call":
                calls.add(identifier)
            ids.add(identifier)
            messages.append({"id": identifier, "role": role, "tokens": item["tokens"], "answers": answers})
        made.append({"name": name, "limit": entry["limit"], "messages": messages})
    return made, None


def trim(conversation):
    """{kept, evicted, total, status} for one conversation."""
    messages = conversation["messages"]
    limit = conversation["limit"]
    order = [message["id"] for message in messages]
    tokens = {message["id"]: message["tokens"] for message in messages}
    total = sum(tokens.values())
    if total <= limit:
        return {"kept": list(order), "evicted": [], "total": total, "status": UNCHANGED}

    safe = {message["id"] for message in messages if message["role"] == "system"}
    users = [message["id"] for message in messages if message["role"] == "user"]
    if users:
        safe.add(users[-1])
    gone, evicted = set(), []
    answer_of = {message["answers"]: message["id"] for message in messages if message["role"] == "tool_result"}

    def take(identifier):
        nonlocal total
        gone.add(identifier)
        evicted.append(identifier)
        total -= tokens[identifier]

    for message in messages:  # (a) tool_results, oldest first
        if total <= limit:
            break
        if message["role"] == "tool_result" and message["id"] not in gone and message["id"] not in safe:
            take(message["id"])
    for message in messages:  # (b) whatever unprotected is left, oldest first
        if total <= limit:
            break
        if message["id"] not in gone and message["id"] not in safe:
            take(message["id"])
            answer = answer_of.get(message["id"])
            if message["role"] == "tool_call" and answer is not None and answer not in gone:
                take(answer)
    return {
        "kept": [identifier for identifier in order if identifier not in gone],
        "evicted": evicted,
        "total": total,
        "status": OVERFLOW if total > limit else TRIMMED,
    }


def expected(conversations):
    """Every conversation's trimming, in the file's order."""
    return [{"conversation": conversation["name"], **trim(conversation)} for conversation in conversations]


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(workspace, name):
    if not isinstance(name, str) or not name:
        return None, f"{name!r} is not a file name"
    relative = Path(name)
    root = Path(workspace).resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root) or path == root:
        return None, f"{name!r} must name a file inside the workspace"
    if not path.is_file():
        return None, f"no file at {name}"
    if path.stat().st_size > 4_000_000:
        return None, f"{name} is too large to read"
    return path.read_text(encoding="utf-8"), None


def load(workspace, name):
    """(the conversations, the source hash, None) or (None, None, what is wrong)."""
    text, problem = read_text(workspace, name)
    if problem:
        return None, None, problem
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as error:
        return None, None, f"{name} is not JSON: {error}"
    conversations, problem = parse_conversations(data)
    if problem:
        return None, None, f"{name}: {problem}"
    return conversations, sha256_text(text), None

"""The eviction policy, twice: the plugin's and the verifier's own, on conversations built to
isolate one rule at a time."""

from __future__ import annotations

import pytest

import verifier.rules as checked
from contract import Status
from plugin.rules import read_conversations_data, trim


def conversation(limit, *messages, name="c"):
    made = []
    for item in messages:
        identifier, role, tokens = item[0], item[1], item[2]
        answers = item[3] if len(item) > 3 else None
        made.append({"id": identifier, "role": role, "tokens": tokens, "answers": answers})
    return {"name": name, "limit": limit, "messages": made}


def both(data):
    """(kept, evicted, total, status) from each implementation, asserted equal."""
    [one] = read_conversations_data({"conversations": [data]})
    mine = trim(one)
    [other] = checked.parse_conversations({"conversations": [data]})[0]
    theirs = checked.trim(other)
    assert (mine.kept, mine.evicted, mine.total, mine.status.value) == (
        theirs["kept"],
        theirs["evicted"],
        theirs["total"],
        theirs["status"],
    ), "the two implementations disagree"
    return mine


def test_exactly_at_the_limit_is_unchanged() -> None:
    made = both(conversation(30, ("s1", "system", 10), ("u1", "user", 20)))
    assert made.status is Status.UNCHANGED and made.evicted == [] and made.total == 30


def test_one_token_over_evicts_once_and_stops() -> None:
    made = both(conversation(49, ("s1", "system", 10), ("c1", "tool_call", 9), ("r1", "tool_result", 11, "c1"), ("u1", "user", 20)))
    assert made.evicted == ["r1"] and made.total == 39 and made.status is Status.TRIMMED


def test_the_last_user_message_is_protected_and_earlier_ones_are_not() -> None:
    made = both(conversation(40, ("u1", "user", 25), ("a1", "assistant", 15), ("u2", "user", 20)))
    assert "u1" in made.evicted and "u2" in made.kept


def test_tool_results_go_before_anything_else() -> None:
    made = both(conversation(60, ("s1", "system", 10), ("a1", "assistant", 20), ("c1", "tool_call", 10), ("r1", "tool_result", 30, "c1"), ("u1", "user", 15)))
    assert made.evicted == ["r1"], "the assistant message is older but a tool_result goes first"
    assert "c1" in made.kept, "the tool_call stays when only its result is evicted"


def test_the_second_pass_takes_the_oldest_unprotected_once_the_results_are_gone() -> None:
    """85 tokens against a limit of 45. Pass (a) takes r1 (15) and leaves 70, still over. Pass (b)
    takes a1 (20) leaving 50, still over, then c1 (10) leaving 40, which fits."""
    made = both(conversation(45, ("s1", "system", 10), ("a1", "assistant", 20), ("c1", "tool_call", 10), ("r1", "tool_result", 15, "c1"), ("a2", "assistant", 18), ("u1", "user", 12)))
    assert made.evicted == ["r1", "a1", "c1"]
    assert made.kept == ["s1", "a2", "u1"] and made.total == 40
    assert made.status is Status.TRIMMED


def test_nothing_extra_is_evicted_once_it_fits() -> None:
    """54 against 40. Pass (a) takes r1 (20), leaving 34, which fits, so a1 is never considered
    even though it is unprotected and would have been next."""
    made = both(conversation(40, ("s1", "system", 10), ("c1", "tool_call", 5), ("r1", "tool_result", 20, "c1"), ("a1", "assistant", 9), ("u1", "user", 10)))
    assert made.evicted == ["r1"] and made.total == 34
    assert "a1" in made.kept and "c1" in made.kept


def test_everything_unprotected_gone_and_still_over_is_overflow() -> None:
    made = both(conversation(20, ("s1", "system", 25), ("a1", "assistant", 10), ("u1", "user", 15)))
    assert made.status is Status.OVERFLOW
    assert made.kept == ["s1", "u1"] and made.total == 40


def test_a_pending_tool_call_is_evicted_alone() -> None:
    made = both(conversation(25, ("s1", "system", 10), ("c1", "tool_call", 14), ("a1", "assistant", 8), ("u1", "user", 7)))
    assert "c1" in made.evicted


def test_rule_3b_is_unreachable_under_these_rules() -> None:
    """Pass (a) evicts every tool_result, and pass (b) only runs when that was not enough, so no
    tool_result can survive into pass (b). The clause that would evict one with its tool_call is
    implemented because the policy states it; this pins that nothing reaches it."""
    data = conversation(1, ("s1", "system", 1), ("c1", "tool_call", 50), ("r1", "tool_result", 50, "c1"), ("a1", "assistant", 50), ("u1", "user", 1))
    made = both(data)
    assert made.evicted.index("r1") < made.evicted.index("c1"), "the result always goes before its call"
    # the call is evicted in pass (b), by which point its result is long gone
    assert made.status is Status.OVERFLOW


@pytest.mark.parametrize(
    "broken",
    [
        {"conversations": []},
        {"conversations": [{"name": "c", "limit": 10, "messages": []}]},
        {"conversations": [{"name": "c", "limit": 10, "messages": [{"id": "a", "role": "nope", "tokens": 1, "answers": None}]}]},
        {"conversations": [{"name": "c", "limit": 10, "messages": [{"id": "a", "role": "user", "tokens": -1, "answers": None}]}]},
        {"conversations": [{"name": "c", "limit": 10, "messages": [{"id": "a", "role": "user", "tokens": 1, "answers": None}, {"id": "a", "role": "user", "tokens": 1, "answers": None}]}]},
        {"conversations": [{"name": "c", "limit": 10, "messages": [{"id": "r", "role": "tool_result", "tokens": 1, "answers": "nope"}]}]},
    ],
)
def test_a_file_that_breaks_the_format_is_refused_by_both(broken: dict) -> None:
    from plugin.rules import RuleError

    found, problem = checked.parse_conversations(broken)
    assert found is None and problem
    with pytest.raises(RuleError):
        read_conversations_data(broken)


def test_the_two_policies_agree_on_random_conversations_and_keep_the_invariants() -> None:
    """1,200 random conversations through both implementations. They must agree, and the answer
    must keep the invariants the policy implies however it was reached: kept and evicted
    partition the messages, no protected message is evicted, the kept list is the conversation's
    order, the total is the kept messages added up, the status matches what actually happened,
    and nothing was evicted that the limit did not need. The invariants are the part that could
    catch a mistake both implementations share, since one hand wrote both."""
    import random

    randomly = random.Random(20261106)
    roles = ["system", "user", "assistant", "tool_call", "tool_result"]
    checked = 0
    for _ in range(1200):
        messages, calls = [], []
        for position in range(randomly.randint(1, 12)):
            role = randomly.choice(roles)
            answers = None
            if role == "tool_result":
                if not calls:
                    role = randomly.choice(roles[:4])
                else:
                    answers = randomly.choice(calls)
            identifier = f"m{position}"
            if role == "tool_call":
                calls.append(identifier)
            messages.append({"id": identifier, "role": role, "tokens": randomly.choice([0, 1, 5, 20, 50, 120]), "answers": answers})
        data = {"name": "c", "limit": randomly.choice([0, 1, 20, 60, 150, 400]), "messages": messages}
        payload = {"conversations": [data]}
        if checked_parse(payload) is None:
            continue
        checked += 1
        made = both(data)  # raises if the two disagree

        ids = [message["id"] for message in messages]
        tokens = {message["id"]: message["tokens"] for message in messages}
        safe = {message["id"] for message in messages if message["role"] == "system"}
        users = [message["id"] for message in messages if message["role"] == "user"]
        if users:
            safe.add(users[-1])
        assert set(made.kept) | set(made.evicted) == set(ids)
        assert not set(made.kept) & set(made.evicted)
        assert safe <= set(made.kept), "a protected message was evicted"
        assert made.kept == [i for i in ids if i in set(made.kept)], "the kept list lost the conversation's order"
        assert made.total == sum(tokens[i] for i in made.kept)
        original = sum(tokens.values())
        assert (made.status.value == "unchanged") == (original <= data["limit"])
        if made.status.value == "trimmed":
            assert made.total <= data["limit"]
            assert made.total + tokens[made.evicted[-1]] > data["limit"], "the last eviction was not needed"
        if made.status.value == "overflow":
            assert set(made.kept) == safe and made.total > data["limit"]
        results = [m["id"] for m in messages if m["role"] == "tool_result"]
        went_results = [i for i in made.evicted if i in results]
        went_others = [i for i in made.evicted if i not in results]
        assert went_results == [i for i in results if i in set(went_results)], "results not oldest first"
        if went_others and went_results:
            assert made.evicted.index(went_others[0]) > made.evicted.index(went_results[-1]), "pass (b) started early"
    assert checked > 800, "most random conversations should be well formed"


def checked_parse(payload):
    """The parsed payload, or None when the random case is not a well-formed one."""
    found, problem = checked.parse_conversations(payload)
    return None if problem else found

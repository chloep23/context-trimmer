"""drop_oldest: held to the cruder rule it claims, not to the task's policy."""

from .checks import result, verify_source_input
from .rules import load

verify_input = verify_source_input


def verify_output(input_value, output_value, workspace):
    conversations, _, problem = load(workspace, input_value.conversations)
    if problem:
        return result(False, problem)
    if [entry.conversation for entry in output_value.results] != [entry["name"] for entry in conversations]:
        return result(False, "the results are not the file's conversations, in its order")
    for given, want in zip(output_value.results, conversations, strict=True):
        messages = want["messages"]
        total = sum(message["tokens"] for message in messages)
        kept = [message["id"] for message in messages]
        for message in messages:
            if total <= want["limit"]:
                break
            kept.remove(message["id"])
            total -= message["tokens"]
        if list(given.kept) != kept or given.total != total:
            return result(False, f"{given.conversation}: dropping the oldest until it fits does not give this")
    return result(True, "the oldest go until it fits", conversations=len(conversations))

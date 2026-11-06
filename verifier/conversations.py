"""read_conversations: every conversation, recomputed from the file."""

from .checks import result, verify_source_input
from .rules import load

verify_input = verify_source_input


def verify_output(input_value, output_value, workspace):
    conversations, source, problem = load(workspace, input_value.conversations)
    if problem:
        return result(False, problem)
    if output_value.source_sha256 != source:
        return result(False, "source_sha256 is not the sha256 of the conversations file")
    if [entry.name for entry in output_value.conversations] != [entry["name"] for entry in conversations]:
        return result(False, "the conversations are not the file's, in its order")
    for given, want in zip(output_value.conversations, conversations, strict=True):
        if given.limit != want["limit"]:
            return result(False, f"{given.name}: the limit is not the file's")
        shown = [(m.id, m.role.value, m.tokens, m.answers) for m in given.messages]
        if shown != [(m["id"], m["role"], m["tokens"], m["answers"]) for m in want["messages"]]:
            return result(False, f"{given.name}: the messages are not the file's, in its order")
    return result(True, "the conversations are the file's", conversations=len(conversations))

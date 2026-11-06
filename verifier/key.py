"""trim_key: the policy applied again."""

from .checks import result, verify_source_input
from .rules import expected, load
from .task import given_trims

verify_input = verify_source_input


def verify_output(input_value, output_value, workspace):
    conversations, source, problem = load(workspace, input_value.conversations)
    if problem:
        return result(False, problem)
    if output_value.source_sha256 != source:
        return result(False, "source_sha256 is not the sha256 of the conversations file")
    wanted = expected(conversations)
    given = given_trims(output_value)
    if given != wanted:
        bad = next((w for w, g in zip(wanted, given, strict=False) if w != g), None)
        where = f" for {bad['conversation']}" if bad else ""
        return result(False, f"the answer key is not what the policy gives{where}")
    return result(True, "the answer key follows the policy", conversations=len(wanted))

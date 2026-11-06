"""report_trims: the call and the merged report it returns."""

from .checks import result, run_folder, source_problem
from .rules import load
from .task import given_trims


def verify_input(value):
    problem = source_problem(value.task)
    if problem:
        return result(False, problem)
    named = [entry.conversation for entry in value.trims]
    if len(set(named)) != len(named):
        return result(False, "a conversation is reported twice in one call")
    for entry in value.trims:
        if entry.total < 0:
            return result(False, f"{entry.conversation}: the total is negative")
    return result(True, "input is valid", trims=len(named))


def verify_output(input_value, output_value, workspace):
    conversations, _, problem = load(workspace, input_value.task.conversations)
    if problem:
        return result(False, problem)
    order = [entry["name"] for entry in conversations]
    report = output_value.report
    if output_value.path != f"{run_folder(report.run)}/trims.json":
        return result(False, f"the report path reads {output_value.path!r}, not this run's")
    if output_value.covered != len(report.trims):
        return result(False, f"covered reads {output_value.covered}, but the report holds {len(report.trims)} conversations")
    if output_value.complete != (len(report.trims) == len(order)):
        return result(False, f"complete reads {output_value.complete}, which is not whether every conversation is reported")
    listed = [entry.conversation for entry in report.trims]
    if listed != [name for name in order if name in set(listed)]:
        return result(False, "the merged report is not in the file's order")
    held = {entry["conversation"]: entry for entry in given_trims(report)}
    for entry in given_trims(input_value):
        if held.get(entry["conversation"]) != entry:
            return result(False, f"{entry['conversation']} is not in the merged report as this call gave it")
    return result(True, "the merged report holds this call", covered=output_value.covered)

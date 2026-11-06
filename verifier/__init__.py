"""The deterministic verifier: each tool's checks live in its own module, on contract instances."""

from contract import Task, Tool

from . import conversations, key, oldest, report, task

TOOLS = {
    Tool.READ_CONVERSATIONS: conversations,
    Tool.REPORT_TRIMS: report,
    Tool.TRIM_KEY: key,
    Tool.DROP_OLDEST: oldest,
}
TASKS = {Task.TRIM_CONTEXT: task.verify_task}


def verify_input(tool, value):
    return TOOLS[Tool(tool)].verify_input(value)


def verify_output(tool, input_value, output_value, workspace):
    return TOOLS[Tool(tool)].verify_output(input_value, output_value, workspace)


def verify_task(task, arguments, output, reference, workspace):
    return TASKS[Task(task)](arguments, output, reference, workspace)

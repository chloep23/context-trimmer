"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    READ_CONVERSATIONS = "read_conversations"
    REPORT_TRIMS = "report_trims"
    TRIM_KEY = "trim_key"
    DROP_OLDEST = "drop_oldest"


class Task(Enum):
    TRIM_CONTEXT = "trim-context"


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class Status(Enum):
    UNCHANGED = "unchanged"
    TRIMMED = "trimmed"
    OVERFLOW = "overflow"


CONVERSATIONS = "The conversations file, relative to the workspace, e.g. conversations.json"
MESSAGE_ID = "The message's id, exactly as the conversation writes it"
NAME = "The conversation's name, exactly as the file writes it"
RUN = "The run that read this task's file, as read_conversations returns it: copy it unchanged"
SOURCE_HASH = "sha256 of the conversations file the trimming was worked out from"


@dataclass
class Message(Output):
    id: Annotated[str, MESSAGE_ID]
    role: Annotated[Role, "system, user, assistant, tool_call or tool_result"]
    tokens: Annotated[int, "How many tokens the message takes in the context"]
    answers: Annotated[str | None, "For a tool_result, the id of the tool_call it answers; null for every other role"]


@dataclass
class Conversation(Output):
    name: Annotated[str, NAME]
    limit: Annotated[int, "The most tokens this conversation's context may hold"]
    messages: Annotated[list[Message], "Every message, oldest first"]


@dataclass
class TrimInput(Output):
    conversations: Annotated[str, CONVERSATIONS]


@dataclass
class ConversationsOutput(Output):
    run: Annotated[str, RUN]
    conversations: Annotated[list[Conversation], "Every conversation in the file, in its order"]
    source_sha256: Annotated[str, SOURCE_HASH]


@dataclass
class Trim(Output):
    conversation: Annotated[str, NAME]
    kept: Annotated[list[str], "The ids still in the context, in the conversation's own order"]
    evicted: Annotated[list[str], "The ids evicted, in the order they were evicted; [] when none was"]
    total: Annotated[int, "The tokens the kept messages add up to"]
    status: Annotated[
        Status,
        "unchanged: nothing was over the limit; trimmed: eviction brought it under; "
        "overflow: everything unprotected is gone and it is still over",
    ]


@dataclass
class ReportInput(Output):
    task: Annotated[TrimInput, "The task input, exactly as given"]
    trims: Annotated[list[Trim], "The conversations this call reports; an earlier entry for the same one is replaced"]


@dataclass
class TrimsOutput(Output):
    task: Annotated[TrimInput, "The task input, exactly as given"]
    run: Annotated[str, RUN]
    trims: Annotated[list[Trim], "One entry per conversation, in the file's order"]


@dataclass
class ReportOutput(Output):
    path: Annotated[str, "The report file, relative to the workspace: reports/<run>/trims.json"]
    covered: Annotated[int, "How many of the conversations the merged report holds so far"]
    complete: Annotated[bool, "Whether every conversation is reported: then submit the report"]
    report: Annotated[TrimsOutput, "The merged report so far"]


@dataclass
class KeyOutput(Output):
    source_sha256: Annotated[str, SOURCE_HASH]
    trims: Annotated[list[Trim], "Every conversation's trimming, in the file's order"]


@dataclass
class Dropped(Output):
    conversation: Annotated[str, NAME]
    kept: Annotated[list[str], "What is left after dropping the oldest messages, whatever their role"]
    total: Annotated[int, "What those add up to"]


@dataclass
class DropOldestOutput(Output):
    results: Annotated[list[Dropped], "One entry per conversation, in the file's order"]


TOOLS = {
    Tool.READ_CONVERSATIONS: (TrimInput, ConversationsOutput),
    Tool.REPORT_TRIMS: (ReportInput, ReportOutput),
    Tool.TRIM_KEY: (TrimInput, KeyOutput),
    Tool.DROP_OLDEST: (TrimInput, DropOldestOutput),
}
TASKS = {Task.TRIM_CONTEXT: (TrimInput, TrimsOutput)}

"""
Pydantic schemas that describe the workspace tools.

These schemas are shared between LangChain tool definitions and the registry
metadata so the CLI can surface structured information for other agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping

from pydantic import BaseModel, Field

__all__ = [
    "DocumentToolInput",
    "DocumentToolOutput",
    "PythonCommandInput",
    "PythonCommandOutput",
    "ShellCommandInput",
    "ShellCommandOutput",
    "TavilySearchInput",
    "TavilySearchOutput",
    "descriptor_schema",
]

DocumentWorkflowKey = Literal["report", "patent_disclosure", "plan", "project_proposal"]


class ShellCommandInput(BaseModel):
    command: str = Field(..., description="Shell command to execute within the workspace.")
    timeout: int | None = Field(None, ge=1, description="Optional timeout override in seconds.")


class ShellCommandOutput(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    cwd: str
    duration: float
    timestamp: float


class PythonCommandInput(BaseModel):
    code: str = Field(..., description="Python source code snippet to execute.")


class PythonCommandOutput(BaseModel):
    code: str
    stdout: str
    stderr: str
    duration: float
    timestamp: float
    timed_out: bool


class TavilySearchInput(BaseModel):
    query: str = Field(..., description="Natural language search query.")
    options: dict[str, Any] | None = Field(default=None, description="Optional Tavily MCP parameters.")


class TavilySearchOutput(BaseModel):
    status: Literal["success", "error"]
    data: Mapping[str, Any] | None = None
    message: str | None = None


class DocumentToolInput(BaseModel):
    workflow: DocumentWorkflowKey = Field(..., description="Document workflow identifier.")
    topic: str = Field(..., description="Topic or subject for the document.")
    instructions: str | None = Field(default=None, description="Additional authoring instructions.")
    audience: str | None = Field(default=None, description="Intended reader description.")
    language: str = Field(default="zh", description="Output language (default: zh).")
    skip_research: bool = Field(default=False, description="Disable Tavily research stage.")


class DocumentToolOutput(BaseModel):
    status: Literal["success", "error"]
    data: Mapping[str, Any] | None = None
    message: str | None = None


@dataclass(slots=True, frozen=True)
class _SchemaPair:
    input_model: type[BaseModel]
    output_model: type[BaseModel]


_DESCRIPTOR_SCHEMAS: Mapping[str, _SchemaPair] = {
    "runtime.shell": _SchemaPair(ShellCommandInput, ShellCommandOutput),
    "runtime.python": _SchemaPair(PythonCommandInput, PythonCommandOutput),
    "research.tavily": _SchemaPair(TavilySearchInput, TavilySearchOutput),
    "docs.report": _SchemaPair(DocumentToolInput, DocumentToolOutput),
    "docs.patent_disclosure": _SchemaPair(DocumentToolInput, DocumentToolOutput),
    "docs.plan": _SchemaPair(DocumentToolInput, DocumentToolOutput),
    "docs.project_proposal": _SchemaPair(DocumentToolInput, DocumentToolOutput),
}


def descriptor_schema(descriptor_name: str) -> Mapping[str, Any] | None:
    schema_pair = _DESCRIPTOR_SCHEMAS.get(descriptor_name)
    if not schema_pair:
        return None
    payload: MutableMapping[str, Any] = {
        "input_schema": schema_pair.input_model.model_json_schema(),
        "output_schema": schema_pair.output_model.model_json_schema(),
    }
    return payload

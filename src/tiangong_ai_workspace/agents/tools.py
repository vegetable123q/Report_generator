"""
LangChain tool definitions that expose workspace capabilities to agents.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from langchain_core.tools import tool

from ..tooling.executors import PythonExecutor, ShellExecutor
from ..tooling.tavily import TavilySearchClient, TavilySearchError
from ..tooling.tool_schemas import (
    DocumentToolInput,
    DocumentToolOutput,
    PythonCommandInput,
    PythonCommandOutput,
    ShellCommandInput,
    ShellCommandOutput,
    TavilySearchInput,
    TavilySearchOutput,
)
from .workflows import DocumentWorkflowConfig, DocumentWorkflowType, run_document_workflow

__all__ = [
    "create_document_tool",
    "create_python_tool",
    "create_shell_tool",
    "create_tavily_tool",
]


def create_shell_tool(executor: Optional[ShellExecutor] = None, *, name: str = "run_shell") -> Any:
    exec_instance = executor or ShellExecutor()

    @tool(name, args_schema=ShellCommandInput)
    def run_shell(command: str, timeout: int | None = None) -> Mapping[str, Any]:
        """Execute a shell command inside the workspace environment."""

        result = exec_instance.run(command, timeout=timeout)
        payload = ShellCommandOutput(**result.to_dict())
        return payload.model_dump()

    return run_shell


def create_python_tool(executor: Optional[PythonExecutor] = None, *, name: str = "run_python") -> Any:
    exec_instance = executor or PythonExecutor()

    @tool(name, args_schema=PythonCommandInput)
    def run_python(code: str) -> Mapping[str, Any]:
        """Execute Python code using the shared workspace interpreter."""

        result = exec_instance.run(code)
        payload = PythonCommandOutput(**result.to_dict())
        return payload.model_dump()

    return run_python


def create_tavily_tool(client: Optional[TavilySearchClient] = None, *, name: str = "tavily_search") -> Any:
    tavily_client = client or TavilySearchClient()

    @tool(name, args_schema=TavilySearchInput)
    def tavily_search(query: str, options: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        """Search the internet using the configured Tavily MCP service."""

        try:
            result = tavily_client.search(query, options=dict(options or {}))
        except TavilySearchError as exc:
            payload = TavilySearchOutput(status="error", message=str(exc))
            return payload.model_dump()
        payload = TavilySearchOutput(status="success", data=result)
        return payload.model_dump()

    return tavily_search


def create_document_tool(*, name: str = "generate_document") -> Any:
    @tool(name, args_schema=DocumentToolInput)
    def generate_document(
        workflow: str,
        topic: str,
        instructions: str | None = None,
        audience: str | None = None,
        language: str = "zh",
        skip_research: bool = False,
    ) -> Mapping[str, Any]:
        """Generate a structured document using the LangGraph workflow."""

        try:
            workflow_type = DocumentWorkflowType(workflow)
        except ValueError:
            payload = DocumentToolOutput(status="error", message=f"Unsupported workflow '{workflow}'.")
            return payload.model_dump()

        config = DocumentWorkflowConfig(
            workflow=workflow_type,
            topic=topic,
            instructions=instructions,
            audience=audience,
            language=language,
            include_research=not skip_research,
        )
        result = run_document_workflow(config)
        payload = DocumentToolOutput(status="success", data=result)
        return payload.model_dump()

    return generate_document

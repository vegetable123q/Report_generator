"""
Document-centric LangGraph workflows tailored for everyday paperwork.

The goal is to keep the orchestration lightweight while still benefiting from
LangChain's prompt tooling and LangGraph's explicit step graph, making it easy
for agents to inspect and reuse the pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

from ..tooling.llm import ModelPurpose, ModelRouter
from ..tooling.tavily import TavilySearchClient, TavilySearchError

TEMPLATES_ROOT = Path(__file__).resolve().parent.parent / "templates"

__all__ = ["DocumentWorkflowConfig", "DocumentWorkflowType", "run_document_workflow"]


class DocumentWorkflowType(str, Enum):
    """Supported document generation workflows."""

    REPORT = "report"
    PATENT_DISCLOSURE = "patent_disclosure"
    PLAN = "plan"
    PROJECT_PROPOSAL = "project_proposal"

    @property
    def template_name(self) -> str:
        return {
            DocumentWorkflowType.REPORT: "report.md",
            DocumentWorkflowType.PATENT_DISCLOSURE: "patent_disclosure.md",
            DocumentWorkflowType.PLAN: "plan.md",
            DocumentWorkflowType.PROJECT_PROPOSAL: "project_proposal.md",
        }[self]

    @property
    def prompt_tone(self) -> str:
        return {
            DocumentWorkflowType.REPORT: "professional and concise",
            DocumentWorkflowType.PATENT_DISCLOSURE: "precise, technical, legally mindful",
            DocumentWorkflowType.PLAN: "clear, action-oriented",
            DocumentWorkflowType.PROJECT_PROPOSAL: "persuasive and outcome-focused",
        }[self]


@dataclass(slots=True)
class DocumentWorkflowConfig:
    """Configuration for running a document workflow."""

    workflow: DocumentWorkflowType
    topic: str
    instructions: Optional[str] = None
    audience: Optional[str] = None
    language: str = "zh"
    include_research: bool = True
    search_query: Optional[str] = None
    research_options: Optional[Mapping[str, Any]] = None
    temperature: float = 0.4
    model_purpose: ModelPurpose = "general"


class DocumentState(TypedDict, total=False):
    topic: str
    instructions: str
    audience: str
    language: str
    tone_hint: str
    template_text: str
    research: Sequence[Mapping[str, Any]]
    research_attachments: Sequence[Mapping[str, Any]]
    outline: str
    draft: str


def run_document_workflow(
    config: DocumentWorkflowConfig,
    *,
    llm: Optional[Runnable] = None,
    tavily: Optional[TavilySearchClient] = None,
) -> Mapping[str, Any]:
    """
    Execute the document workflow and return structured results.

    The function compiles a LangGraph `StateGraph` and then invokes it with the
    configuration provided. Callers can inject a custom `Runnable` to mock the
    language model during tests.
    """

    router = ModelRouter() if llm is None else None
    model = llm or router.create_chat_model(purpose=config.model_purpose, temperature=config.temperature)

    template_text = _load_template(config.workflow)
    if tavily is not None:
        research_client = tavily
    elif config.include_research:
        research_client = TavilySearchClient()
    else:
        research_client = None

    graph = StateGraph(DocumentState)

    def research_node(state: DocumentState) -> DocumentState:
        if not config.include_research or research_client is None:
            return state
        query = config.search_query or config.topic
        try:
            result = research_client.search(query, options=config.research_options)
        except TavilySearchError as exc:
            # Attach the error to the state so the workflow can proceed without research data.
            return {
                **state,
                "research": [{"summary": "Research step failed", "details": str(exc)}],
                "research_attachments": [],
            }

        research_payload = _normalise_research(result.get("result"))
        attachments = _normalise_attachments(result.get("attachments"))
        return {
            **state,
            "research": research_payload,
            "research_attachments": attachments,
        }

    def outline_node(state: DocumentState) -> DocumentState:
        prompt = _build_outline_prompt()
        chain = prompt | model | StrOutputParser()
        research_text = _summarise_research(state.get("research", []))
        instructions = state.get("instructions", "")
        response = chain.invoke(
            {
                "topic": state["topic"],
                "audience": state.get("audience") or "general stakeholders",
                "language": state.get("language") or "zh",
                "tone": state.get("tone_hint"),
                "instructions": instructions,
                "template": template_text,
                "research": research_text,
            }
        )
        return {**state, "outline": response}

    def draft_node(state: DocumentState) -> DocumentState:
        prompt = _build_draft_prompt()
        chain = prompt | model | StrOutputParser()
        response = chain.invoke(
            {
                "topic": state["topic"],
                "outline": state.get("outline", ""),
                "language": state.get("language") or "zh",
                "tone": state.get("tone_hint"),
                "audience": state.get("audience") or "general stakeholders",
                "instructions": state.get("instructions", ""),
                "template": template_text,
                "research": _summarise_research(state.get("research", [])),
            }
        )
        return {**state, "draft": response}

    graph.add_node("research", research_node)
    graph.add_node("outline", outline_node)
    graph.add_node("draft", draft_node)

    if config.include_research:
        graph.set_entry_point("research")
        graph.add_edge("research", "outline")
    else:
        graph.set_entry_point("outline")
    graph.add_edge("outline", "draft")
    graph.add_edge("draft", END)

    app = graph.compile()
    initial_state: DocumentState = {
        "topic": config.topic,
        "instructions": config.instructions or "",
        "audience": config.audience or "",
        "language": config.language,
        "tone_hint": config.workflow.prompt_tone,
        "template_text": template_text,
        "research": [],
        "research_attachments": [],
    }

    final_state = app.invoke(initial_state)
    return {
        "workflow": config.workflow.value,
        "topic": config.topic,
        "outline": final_state.get("outline", ""),
        "draft": final_state.get("draft", ""),
        "research": final_state.get("research", []),
        "attachments": final_state.get("research_attachments", []),
        "language": config.language,
        "audience": config.audience,
    }


def _load_template(workflow: DocumentWorkflowType) -> str:
    template_path = TEMPLATES_ROOT / workflow.template_name
    if not template_path.exists():
        return ""
    return template_path.read_text(encoding="utf-8")


def _build_outline_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ("You are an expert documentation strategist. " "Draft a structured outline based on the provided topic, instructions, research, and template excerpts."),
            ),
            (
                "human",
                (
                    "Topic: {topic}\n"
                    "Target audience: {audience}\n"
                    "Preferred language: {language}\n"
                    "Tone guidance: {tone}\n"
                    "Specific instructions: {instructions}\n"
                    "Template guidance:\n{template}\n"
                    "Research summary:\n{research}\n"
                    "Return a markdown outline with clear headings and bullet points."
                ),
            ),
        ]
    )


def _build_draft_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                ("You are a senior documentation specialist. " "Expand the outline into a polished draft that is ready for human review."),
            ),
            (
                "human",
                (
                    "Topic: {topic}\n"
                    "Target audience: {audience}\n"
                    "Preferred language: {language}\n"
                    "Tone guidance: {tone}\n"
                    "Specific instructions: {instructions}\n"
                    "Research summary:\n{research}\n"
                    "Starting outline:\n{outline}\n"
                    "Template guidance:\n{template}\n"
                    "Write a complete draft in markdown."
                ),
            ),
        ]
    )


def _normalise_research(result: Any) -> Sequence[Mapping[str, Any]]:
    if result is None:
        return []
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        normalised = []
        for item in result:
            if isinstance(item, Mapping):
                normalised.append(dict(item))
            else:
                normalised.append({"summary": str(item)})
        return normalised
    if isinstance(result, Mapping):
        return [dict(result)]
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return _normalise_research(parsed)
        except json.JSONDecodeError:
            return [{"summary": result}]
    return [{"summary": str(result)}]


def _normalise_attachments(attachments: Any) -> Sequence[Mapping[str, Any]]:
    if not attachments:
        return []
    if isinstance(attachments, Sequence):
        return [dict(item) if isinstance(item, Mapping) else {"value": str(item)} for item in attachments]
    if isinstance(attachments, Mapping):
        return [dict(attachments)]
    return [{"value": str(attachments)}]


def _summarise_research(research: Sequence[Mapping[str, Any]]) -> str:
    if not research:
        return "无公开资料或暂未执行网络检索。"
    lines = []
    for idx, item in enumerate(research, start=1):
        summary = item.get("summary") or item.get("title") or item.get("text") or ""
        url = item.get("url") or item.get("link")
        note = item.get("notes") or item.get("excerpt")
        segment = f"{idx}. {summary}"
        if url:
            segment += f" ({url})"
        if note:
            segment += f"\n   - {note}"
        lines.append(segment.strip())
    return "\n".join(lines)

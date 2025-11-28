"""
Utility helpers for Tiangong AI Workspace tooling.

This package exposes lightly opinionated building blocks such as response
schemas, tool registries, and external service wrappers that agents can reuse.
"""

from .dify import DifyKnowledgeBaseClient
from .embeddings import OpenAICompatibleEmbeddingClient
from .executors import PythonExecutor, ShellExecutor
from .neo4j import Neo4jClient
from .registry import ToolDescriptor, list_registered_tools
from .responses import ResponsePayload, WorkspaceResponse

__all__ = [
    "DifyKnowledgeBaseClient",
    "OpenAICompatibleEmbeddingClient",
    "PythonExecutor",
    "ResponsePayload",
    "Neo4jClient",
    "ShellExecutor",
    "WorkspaceResponse",
    "ToolDescriptor",
    "list_registered_tools",
]

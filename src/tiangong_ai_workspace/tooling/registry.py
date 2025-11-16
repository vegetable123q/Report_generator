"""
In-process tool registry for agent discoverability.

The registry keeps a lightweight catalogue of workflows and integrations so that
agents (and humans) can quickly inspect what the workspace exposes without
reading source code. CLI commands use this module for structured listings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Tuple

from .config import RegistryEntryConfig, load_workspace_config
from .tool_schemas import descriptor_schema

__all__ = ["ToolDescriptor", "register_tool", "list_registered_tools"]


@dataclass(slots=True, frozen=True)
class ToolDescriptor:
    """Metadata describing an agent-facing workflow or integration."""

    name: str
    description: str
    category: str
    entrypoint: str
    tags: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None


_TOOL_REGISTRY: MutableMapping[str, ToolDescriptor] = {}
_BOOTSTRAPPED = False


def register_tool(descriptor: ToolDescriptor) -> None:
    """Register a tool descriptor, replacing any existing entry with the same name."""
    _TOOL_REGISTRY[descriptor.name] = descriptor


def register_many(descriptors: Iterable[ToolDescriptor]) -> None:
    """Bulk register multiple descriptors."""
    for descriptor in descriptors:
        register_tool(descriptor)


def list_registered_tools() -> Mapping[str, ToolDescriptor]:
    """Return an immutable view of the current tool registry."""
    _bootstrap_registry()
    return dict(_TOOL_REGISTRY)


def _bootstrap_registry() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    config = load_workspace_config()
    descriptors = [_convert_entry(entry) for entry in config.registry]
    register_many(descriptors)
    _BOOTSTRAPPED = True


def _convert_entry(entry: RegistryEntryConfig) -> ToolDescriptor:
    metadata: MutableMapping[str, Any] = {}
    if entry.metadata:
        metadata.update(entry.metadata)
    schema = descriptor_schema(entry.name)
    if schema:
        metadata.update(schema)
    return ToolDescriptor(
        name=entry.name,
        description=entry.description,
        category=entry.category,
        entrypoint=entry.entrypoint,
        tags=entry.tags,
        metadata=metadata or None,
    )


_bootstrap_registry()

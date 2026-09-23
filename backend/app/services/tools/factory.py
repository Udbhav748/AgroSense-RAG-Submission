"""Factory for the dynamic tool registry — the single place that decides
which tools are available to agents. Registering a new capability is
adding one tool class here; nothing else in the agent path needs to know
about it (tools are looked up by name at run time)."""

from __future__ import annotations

from app.services.tools.implementations import (
    DiagnoseTool,
    RetrievalTool,
    SummarizationTool,
    VisionQATool,
    WebSearchTool,
)
from app.services.tools.registry import ToolRegistry


def build_tool_registry() -> ToolRegistry:
    """Construct a ToolRegistry with every built-in tool registered."""
    registry = ToolRegistry()
    registry.register(RetrievalTool())
    registry.register(WebSearchTool())
    registry.register(SummarizationTool())
    registry.register(DiagnoseTool())
    registry.register(VisionQATool())
    return registry

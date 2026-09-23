"""Dynamic tool registry package: tools/ defines the agent-facing
invocation surface — tools callable by name with validated arguments,
alongside the inline @track_tool path used by ChatService.

Layout:
- base.py            — BaseTool / ToolContext / ToolResult / ToolSpec
- registry.py        — ToolRegistry (register / get / list / execute)
- implementations.py — the concrete tools
- factory.py         — build_tool_registry()
"""

from app.services.tools.base import BaseTool, ToolContext, ToolResult, ToolSpec
from app.services.tools.registry import ToolNotFoundError, ToolRegistry

__all__ = [
    "BaseTool",
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "ToolRegistry",
    "ToolNotFoundError",
]

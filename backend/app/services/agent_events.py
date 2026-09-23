"""Shared structured logging for agent lifecycle events.

The multi-agent features (router_agent.py, research_agent.py) emit these
events so the checklist's collaboration metrics are computable from the
log the same way Tool Success Rate and Retry Success Rate are:

- agent_started:    a named agent began work on a task.
- agent_completed:  the agent finished, with its outcome and duration.
- agent_handoff:    control passed from one agent to another.

Handoff Accuracy and Agent Idle Time are directly derivable: a handoff
event names both ends (from/to); idle time is the gap between one agent's
agent_completed and the next agent_started with the same request_id.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _emit(event: str, **fields: Any) -> None:
    logger.info(event, extra={"extra_fields": fields})


def log_agent_started(agent: str, task: str, **extra: Any) -> None:
    """Record that `agent` began working on `task` (short shape, not
    content — see _summarize_input's philosophy in tool_registry.py)."""
    _emit("agent_started", agent=agent, task_length=len(task), **extra)


def log_agent_completed(agent: str, task: str, start: float, *, outcome: str, **extra: Any) -> None:
    """Record that `agent` finished. start is a time.perf_counter() value
    captured at the matching log_agent_started; duration is computed here
    so the two events can't drift on clock source."""
    _emit(
        "agent_completed",
        agent=agent,
        task_length=len(task),
        outcome=outcome,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
        **extra,
    )


def log_agent_handoff(frm: str, to: str, task: str, **extra: Any) -> None:
    """Record that `frm` handed `task` off to `to` — the from/to pair is
    what Handoff Accuracy is computed from. Also bumps the
    agent_handoffs_total Prometheus metric the agent-metrics dashboard's
    handoff breakdown panel reads."""
    _emit("agent_handoff", from_agent=frm, to_agent=to, task_length=len(task), **extra)
    from app.core.metrics import get_metrics

    get_metrics().record_agent_handoff(frm=frm, to=to)

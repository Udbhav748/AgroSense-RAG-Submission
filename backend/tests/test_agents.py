"""Tests for the multi-agent layer: the LLM router agent and the Research
agent (the plan→search→read→synthesize specialist that owns the weak/
insufficient-retrieval path).

Covers, for the router: deterministic fast paths short-circuit without an
LLM call, LLM-routed decisions (including the new "research" action),
and failure-safe fallback to the keyword planner on any LLM/parse failure.

For the research agent: the web-search-disabled and human-approval gates
degrade to empty findings without searching, a full run plans→searches→
reads→synthesizes, and network fetch is isolated behind
research_agent._fetch_page_text (monkeypatched here — no real HTTP).

All agent lifecycle events (agent_started / agent_completed /
agent_handoff) are asserted via caplog on app.services.agent_events,
since those events are what Handoff Accuracy / Agent Idle Time derive
from.
"""

import time
from types import SimpleNamespace

import app.services.research_agent as research_agent
from app.core.config import settings
from app.models.document import WebSearchResult
from app.services.research_agent import ResearchAgent
from app.services.router_agent import RouterAgent


class FakeLLM:
    """generate/generate_structured pop from `responses` in call order
    (first the router/plan JSON, then the synthesis answer), or raise."""

    def __init__(self, responses=None, raise_on_call=False):
        self._responses = list(responses) if responses else []
        self._raise = raise_on_call
        self.calls = []

    def generate(self, prompt):
        self.calls.append(("generate", prompt))
        return self._pop()

    def generate_structured(self, prompt):
        self.calls.append(("generate_structured", prompt))
        return self._pop()

    def _pop(self):
        if self._raise:
            raise RuntimeError("llm down")
        return self._responses.pop(0) if self._responses else ""


def stub_planner(query, history=None):
    """Stand-in for ChatService._plan: conversational for greetings,
    summarize-with-uuid for the fixture id, else retrieve."""
    if query.lower().startswith("hi"):
        return SimpleNamespace(action="conversational", document_id=None)
    if "summarize" in query.lower() and "uuid-1" in query:
        return SimpleNamespace(action="summarize", document_id="uuid-1")
    return SimpleNamespace(action="retrieve", document_id=None)


def make_router(llm=None):
    return RouterAgent(llm or FakeLLM(), fallback_planner=stub_planner)


def make_web_result(url="https://example.com/a", snippet="snippet text"):
    return WebSearchResult(title="Example", url=url, snippet=snippet)


class TestRouterFastPaths:
    def test_conversational_decided_without_llm_call(self):
        llm = FakeLLM()
        decision = make_router(llm).decide("hi there")
        assert decision.action == "conversational"
        assert llm.calls == []  # fast path never touched the LLM

    def test_summarize_with_uuid_trusted_without_llm_call(self):
        llm = FakeLLM()
        decision = make_router(llm).decide("please summarize uuid-1")
        assert decision.action == "summarize"
        assert decision.document_id == "uuid-1"
        assert llm.calls == []


class TestRouterLLM:
    def test_research_action_accepted(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", True)
        llm = FakeLLM([r'{"action": "research", "document_id": null, "reason": "current info"}'])
        decision = make_router(llm).decide("what is the latest price of tomatoes?")
        assert decision.action == "research"
        assert decision.document_id is None

    def test_retrieve_action_passthrough(self):
        llm = FakeLLM([r'{"action": "retrieve", "document_id": null, "reason": "corpus question"}'])
        decision = make_router(llm).decide("what does the PMP doc say about scope?")
        assert decision.action == "retrieve"

    def test_summarize_without_id_downgraded_to_retrieve(self):
        llm = FakeLLM([r'{"action": "summarize", "document_id": null, "reason": "wants summary"}'])
        decision = make_router(llm).decide("summarize it for me")
        assert decision.action == "retrieve"

    def test_research_degraded_when_web_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", False)
        llm = FakeLLM([r'{"action": "research", "document_id": null, "reason": "x"}'])
        decision = make_router(llm).decide("any question")
        assert decision.action == "retrieve"


class TestRouterFallback:
    def test_llm_failure_falls_back_to_planner(self):
        llm = FakeLLM(raise_on_call=True)
        decision = make_router(llm).decide("What is a project in the PMP doc?")
        assert decision.action == "retrieve"  # stub_planner's answer

    def test_unknown_action_falls_back(self):
        llm = FakeLLM([r'{"action": "hack", "document_id": null, "reason": "x"}'])
        decision = make_router(llm).decide("anything")
        assert decision.action == "retrieve"

    def test_unparseable_json_falls_back(self):
        llm = FakeLLM(["sure thing, here you go!"])
        decision = make_router(llm).decide("anything")
        assert decision.action == "retrieve"


class TestRouterEvents:
    def test_lifecycle_events_logged(self, caplog, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", True)
        llm = FakeLLM([r'{"action": "research", "document_id": null, "reason": "r"}'])
        with caplog.at_level("INFO", logger="app.services.agent_events"):
            make_router(llm).decide("latest news about floods")
        messages = [r.message for r in caplog.records]
        assert "agent_started" in messages
        assert "agent_completed" in messages
        completed = [r for r in caplog.records if r.message == "agent_completed"][0]
        assert completed.extra_fields["agent"] == "router"
        assert completed.extra_fields["outcome"] == "llm"


class TestResearchAgentGates:
    def test_empty_findings_when_web_search_disabled(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "web_search_enabled", False)
        agent = ResearchAgent(FakeLLM())
        with caplog.at_level("INFO", logger="app"):
            findings = agent.run("any question")
        assert findings.answer == ""
        assert findings.queries == []
        assert "research_degraded_web_disabled" in [r.message for r in caplog.records]

    def test_empty_findings_pending_approval(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", True)
        monkeypatch.setattr(settings, "web_search_requires_approval", True)
        agent = ResearchAgent(FakeLLM())
        findings = agent.run("any question", confirm_web_search=False)
        assert findings.answer == ""


class TestResearchAgentRun:
    def test_full_plan_search_read_synthesize(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", True)
        monkeypatch.setattr(settings, "research_max_subqueries", 3)
        monkeypatch.setattr(settings, "research_read_limit", 2)

        def fake_search(query, max_results=None):
            return [make_web_result(url=f"https://example.com/{len(query)}", snippet=f"snippet for {query}")]

        def fake_fetch(url, max_chars=None):
            return "PAGE CONTENT WITH THE ANSWER. " * 10

        monkeypatch.setattr("app.services.research_agent.search_web", fake_search)
        monkeypatch.setattr("app.services.research_agent._fetch_page_text", fake_fetch)

        # First call is the plan (generate_structured), second is synthesis (generate).
        llm = FakeLLM([
            '{"queries": ["tomato price", "tomato market"]}',
            "Based on the sources, tomatoes cost about $3 a kilo.",
        ])
        findings = ResearchAgent(llm).run("how much do tomatoes cost?")

        assert findings.queries == ["tomato price", "tomato market"]
        assert findings.pages_read == 2
        assert "tomatoes cost about $3 a kilo" in findings.answer
        # The read page text replaced the snippet for pages we fetched.
        assert findings.results[0].snippet.startswith("PAGE CONTENT")
        # Plan + synthesis both hit the LLM.
        assert [call[0] for call in llm.calls] == ["generate_structured", "generate"]

    def test_empty_results_returns_empty_findings(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", True)

        def fake_search(query, max_results=None):
            return []

        monkeypatch.setattr("app.services.research_agent.search_web", fake_search)
        findings = ResearchAgent(FakeLLM([r'{"queries": ["q"]}'])).run("anything")
        assert findings.answer == ""

    def test_plan_parse_failure_defaults_to_original_query(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", True)
        monkeypatch.setattr("app.services.research_agent.search_web", lambda q, max_results=None: [])
        findings = ResearchAgent(FakeLLM(["not json at all"])).run("something")
        assert findings.queries == ["something"]

    def test_unfetchable_page_skipped(self, monkeypatch):
        monkeypatch.setattr(settings, "web_search_enabled", True)
        monkeypatch.setattr(
            "app.services.research_agent.search_web",
            lambda q, max_results=None: [make_web_result()],
        )
        monkeypatch.setattr("app.services.research_agent._fetch_page_text", lambda url, max_chars=None: "")
        llm = FakeLLM([r'{"queries": ["q"]}', "answer"])
        findings = ResearchAgent(llm).run("question")
        assert findings.pages_read == 0
        assert findings.results[0].snippet == "snippet text"  # page skipped, snippet kept


class TestResearchAgentBudget:
    """The total-timeout budget (research_total_timeout_seconds): once a
    deadline has passed, _search/_read stop issuing new network calls
    rather than running the full sequential chain unbounded."""

    def test_search_stops_when_budget_already_exceeded(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            research_agent, "search_web", lambda q, max_results=None: called.append(q) or []
        )
        agent = ResearchAgent(FakeLLM())
        results = agent._search(["q1", "q2"], deadline=time.perf_counter() - 1)
        assert results == []
        assert called == []  # deadline already past — no search issued

    def test_read_stops_when_budget_already_exceeded(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            research_agent, "_fetch_page_text", lambda url, max_chars=None: called.append(url) or "text"
        )
        agent = ResearchAgent(FakeLLM())
        read, count = agent._read([make_web_result()], deadline=time.perf_counter() - 1)
        assert read == []
        assert count == 0
        assert called == []  # deadline already past — no fetch issued


class TestResearchFetchPageText:
    def test_non_http_url_rejected(self):
        from app.services.research_agent import _fetch_page_text

        assert _fetch_page_text("file:///etc/passwd") == ""
        assert _fetch_page_text("javascript:alert(1)") == ""

    def test_private_and_loopback_hosts_rejected(self):
        """IP literals resolve locally (no real DNS/network call), so this
        stays offline while covering the SSRF guard directly."""
        from app.services.research_agent import _fetch_page_text

        assert _fetch_page_text("http://127.0.0.1/secret") == ""
        assert _fetch_page_text("http://10.0.0.5/") == ""
        assert _fetch_page_text("http://169.254.169.254/latest/meta-data/") == ""

    def test_redirect_to_private_host_rejected(self, monkeypatch):
        """A host that passes the initial check but redirects to a private
        address must still be rejected — the guard re-validates every hop,
        not just the URL the agent started with."""
        hosts_ok = {"example.com": True, "127.0.0.1": False}
        monkeypatch.setattr(research_agent, "_is_public_host", lambda h: hosts_ok.get(h, False))

        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            return SimpleNamespace(status_code=302, headers={"location": "http://127.0.0.1/admin"}, text="")

        monkeypatch.setattr(research_agent.httpx, "get", fake_get)

        assert research_agent._fetch_page_text("http://example.com/a") == ""
        assert calls == ["http://example.com/a"]  # redirect target never fetched

    def test_redirect_to_public_host_followed(self, monkeypatch):
        monkeypatch.setattr(research_agent, "_is_public_host", lambda h: True)

        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            if url == "http://example.com/a":
                return SimpleNamespace(status_code=301, headers={"location": "https://example.com/b"}, text="")
            return SimpleNamespace(
                status_code=200, headers={}, text="<p>hello world</p>", raise_for_status=lambda: None
            )

        monkeypatch.setattr(research_agent.httpx, "get", fake_get)

        text = research_agent._fetch_page_text("http://example.com/a")
        assert "hello world" in text
        assert calls == ["http://example.com/a", "https://example.com/b"]

    def test_too_many_redirects_rejected(self, monkeypatch):
        monkeypatch.setattr(research_agent, "_is_public_host", lambda h: True)

        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            return SimpleNamespace(status_code=302, headers={"location": "http://example.com/next"}, text="")

        monkeypatch.setattr(research_agent.httpx, "get", fake_get)

        assert research_agent._fetch_page_text("http://example.com/a") == ""
        assert len(calls) == research_agent._MAX_REDIRECTS + 1

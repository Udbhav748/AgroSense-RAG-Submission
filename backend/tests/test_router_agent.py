"""Unit tests for crop context extraction, RouterAgent's crop/collection
preservation, and agronomy prompt persona.

(This file used to also cover a duplicate query-planning implementation in
app.services.rag.router/reflection_engine — build_diagnosis_query,
plan_query, route_query, ReflectionEngine, verify_chemical_safety — that
had zero production callers; rag_service.py always used its own
equivalents. Removed once Phase 1's agent-graph audit confirmed that (see
git history); extract_crop_context is the one function from that package
that's actually live, imported into rag_service.py.)
"""

from app.models.document import RetrievedChunk
from app.services.prompt_builder import (
    AGRONOMY_PERSONA,
    PERSONAS,
    build_prompt,
)
from app.services.rag.router import extract_crop_context
from app.services.router_agent import RouterAgent


class FakeLLMClient:
    def __init__(self, response: str = "retrieve"):
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


def _fallback_planner_with_crop(query, history=None):
    """Stand-in for ChatService._plan, for testing RouterAgent's crop/
    collection preservation in isolation: extracts crop the same way the
    real planner does (via extract_crop_context) without needing a full
    ChatService instance."""
    from types import SimpleNamespace

    crop = extract_crop_context(query)
    action = "conversational" if query.strip().lower() == "hello" else "retrieve"
    return SimpleNamespace(action=action, document_id=None, crop=crop, collection=crop)


class TestCropExtraction:
    def test_extract_crop_simple(self):
        assert extract_crop_context("How do I manage early blight on tomato?") == "tomato"
        assert extract_crop_context("What is the fungicide for peach scab?") == "peach"
        assert extract_crop_context("Tell me about potato late blight control") == "potato"
        assert extract_crop_context("Is corn leaf spot dangerous?") == "corn"
        assert extract_crop_context("My strawberry leaves have brown spots") == "strawberry"
        assert extract_crop_context("Cedar apple rust treatment") == "apple"

    def test_extract_crop_case_insensitive_and_punctuation(self):
        assert extract_crop_context("TOMATOES: yellow curling leaves!") == "tomato"
        assert extract_crop_context("Treating cherries with powdery mildew.") == "cherry"
        assert extract_crop_context("Bell pepper anthracnose symptoms") == "bell pepper"

    def test_extract_crop_none_when_unrelated(self):
        assert extract_crop_context("What is our Q3 revenue forecast?") is None
        assert extract_crop_context("Hello there!") is None
        assert extract_crop_context(None) is None


class TestRouterAgentCropPreservation:
    def test_router_agent_decide_extracts_crop_and_collection(self):
        router = RouterAgent(
            FakeLLMClient(response="retrieve"), fallback_planner=_fallback_planner_with_crop
        )
        decision = router.decide("What is causing dark spots on my potato leaves?")
        assert decision.action == "retrieve"
        assert decision.crop == "potato"
        assert decision.collection == "potato"

    def test_router_agent_llm_path_preserves_crop(self):
        llm = FakeLLMClient(response="retrieve")
        router = RouterAgent(llm, fallback_planner=_fallback_planner_with_crop)
        decision = router.decide("What is the harvest schedule for sweet corn?")
        assert decision.crop == "corn"
        assert decision.collection == "corn"


class TestAgronomyPromptPersona:
    def test_agronomy_persona_registered_in_personas(self):
        assert "agronomist" in PERSONAS
        assert "agronomy" in PERSONAS
        assert "plant_pathologist" in PERSONAS
        assert "diagnosis" in PERSONAS
        assert PERSONAS["agronomist"] == AGRONOMY_PERSONA

    def test_agronomy_persona_contains_all_six_sections(self):
        sections = [
            "1. **Visual Diagnosis & Severity Assessment**",
            "2. **Field Protocol & Maintenance Schedule**",
            "3. **Organic & Biological Control Remedies (OMRI approved options)**",
            "4. **Chemical Controls & Dosage Protocols**",
            "5. **Long-Term Cultural Practices & Field Sanitation**",
            "6. **Grounded University Extension Citations**",
        ]
        for section in sections:
            assert section in AGRONOMY_PERSONA

    def test_agronomy_persona_contains_safety_mandate(self):
        assert "SAFETY MANDATE" in AGRONOMY_PERSONA
        assert "PPE" in AGRONOMY_PERSONA
        assert "REI" in AGRONOMY_PERSONA
        assert "PHI" in AGRONOMY_PERSONA

    def test_build_prompt_with_agronomy_persona(self):
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text="Chlorothalonil is effective for early blight control at 1.5 pt/acre.",
                score=0.92,
                metadata={"title": "Tomato Disease Guide"},
            )
        ]
        prompt = build_prompt(
            "How to treat tomato early blight?",
            chunks=chunks,
            persona="agronomist",
        )
        assert "Plant Pathology & Agronomy Expert Persona" in prompt
        assert "Field Protocol & Maintenance Schedule" in prompt
        assert "Chlorothalonil" in prompt



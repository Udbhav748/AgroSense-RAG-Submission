"""Unit tests for Multilingual Agricultural Localization & Prompt Construction in AgroSense-RAG."""

import unittest
from unittest.mock import MagicMock

from app.models.document import RetrievedChunk
from app.services.prompt_builder import (
    SUPPORTED_LANGUAGES,
    build_prompt,
    build_structured_prompt,
    get_language_instruction,
)
from app.services.rag_service import ChatService


def make_chunk(text: str = "Apply copper hydroxide for bacterial leaf spot.", doc_id: str = "doc1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id=doc_id,
        text=text,
        score=0.92,
        metadata={"crop": "tomato", "disease": "bacterial spot"},
    )


class TestMultilingualPromptBuilder(unittest.TestCase):
    def test_supported_languages_contains_all_six(self):
        expected = {"en", "es", "hi", "pt", "fr", "sw"}
        self.assertEqual(set(SUPPORTED_LANGUAGES.keys()), expected)

    def test_english_and_none_do_not_inject_foreign_instruction(self):
        chunk = make_chunk()
        prompt_en = build_prompt("How to treat tomato blight?", [chunk], language="en")
        self.assertNotIn("LANGUAGE LOCALIZATION MANDATE", prompt_en)

        prompt_default = build_prompt("How to treat tomato blight?", [chunk])
        self.assertNotIn("LANGUAGE LOCALIZATION MANDATE", prompt_default)

        self.assertIsNone(get_language_instruction("en"))
        self.assertIsNone(get_language_instruction("en-US"))
        self.assertIsNone(get_language_instruction(None))

    def test_spanish_localization_prompt(self):
        chunk = make_chunk()
        prompt = build_prompt("Tratamiento para tizón temprano", [chunk], language="es")
        self.assertIn("LANGUAGE LOCALIZATION MANDATE (Spanish / Español)", prompt)
        self.assertIn("Spanish (Español)", prompt)
        self.assertIn("visual diagnosis", prompt)
        self.assertIn("organic remedies", prompt)
        self.assertIn("cultural practices", prompt)
        self.assertIn("24h emergency", prompt)
        self.assertIn("scientific chemical active ingredient names", prompt)
        self.assertIn("hidróxido de cobre", prompt)

    def test_hindi_localization_prompt(self):
        chunk = make_chunk()
        prompt = build_prompt("टमाटर में झुलसा रोग का उपचार", [chunk], language="hi")
        self.assertIn("LANGUAGE LOCALIZATION MANDATE (Hindi / हिन्दी)", prompt)
        self.assertIn("Hindi (हिन्दी)", prompt)
        self.assertIn("visual diagnosis", prompt)
        self.assertIn("organic remedies", prompt)
        self.assertIn("cultural practices", prompt)
        self.assertIn("24h emergency", prompt)
        self.assertIn("scientific chemical active ingredient names", prompt)
        self.assertIn("मैंकोजेब", prompt)

    def test_portuguese_localization_prompt(self):
        chunk = make_chunk()
        prompt = build_prompt("Como tratar requeima do tomateiro?", [chunk], language="pt")
        self.assertIn("LANGUAGE LOCALIZATION MANDATE (Portuguese / Português)", prompt)
        self.assertIn("Portuguese (Português)", prompt)
        self.assertIn("visual diagnosis", prompt)
        self.assertIn("organic remedies", prompt)
        self.assertIn("cultural practices", prompt)
        self.assertIn("24h emergency", prompt)
        self.assertIn("scientific chemical active ingredient names", prompt)
        self.assertIn("clorotalonil", prompt)

    def test_french_localization_prompt(self):
        chunk = make_chunk()
        prompt = build_prompt("Traitement contre le mildiou de la tomate", [chunk], language="fr")
        self.assertIn("LANGUAGE LOCALIZATION MANDATE (French / Français)", prompt)
        self.assertIn("French (Français)", prompt)
        self.assertIn("visual diagnosis", prompt)
        self.assertIn("organic remedies", prompt)
        self.assertIn("cultural practices", prompt)
        self.assertIn("24h emergency", prompt)
        self.assertIn("scientific chemical active ingredient names", prompt)
        self.assertIn("mancozèbe", prompt)

    def test_swahili_localization_prompt(self):
        chunk = make_chunk()
        prompt = build_prompt("Jinsi ya kutibu ukungu kwenye nyanya", [chunk], language="sw")
        self.assertIn("LANGUAGE LOCALIZATION MANDATE (Swahili / Kiswahili)", prompt)
        self.assertIn("Swahili (Kiswahili)", prompt)
        self.assertIn("visual diagnosis", prompt)
        self.assertIn("organic remedies", prompt)
        self.assertIn("cultural practices", prompt)
        self.assertIn("24h emergency", prompt)
        self.assertIn("scientific chemical active ingredient names", prompt)
        self.assertIn("dawa ya ukungu", prompt)

    def test_structured_prompt_multilingual(self):
        chunk = make_chunk()
        prompt = build_structured_prompt("Tratamiento en español", [chunk], language="es")
        self.assertIn("LANGUAGE LOCALIZATION MANDATE (Spanish / Español)", prompt)
        self.assertIn('"answer":', prompt)
        self.assertIn("Answer (JSON only):", prompt)


class TestMultilingualRAGService(unittest.TestCase):
    def test_handle_query_passes_language_to_prompt(self):
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [make_chunk()]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Diagnóstico en español [1]."

        service = ChatService(
            vector_store=mock_vector_store,
            llm_client=mock_llm,
        )

        with unittest.mock.patch("app.services.rag_service.retrieve", return_value=[make_chunk()]):
            response = service.handle_query(
                "¿Cómo controlar tizón temprano en papa?",
                language="es",
            )

        self.assertEqual(response.answer, "Diagnóstico en español [1].")
        called_prompt = mock_llm.generate.call_args[0][0]
        self.assertIn("LANGUAGE LOCALIZATION MANDATE (Spanish / Español)", called_prompt)

    def test_handle_diagnose_passes_language_to_prompt(self):
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [make_chunk()]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "हिंदी में उपचार निर्देश [1]."

        service = ChatService(
            vector_store=mock_vector_store,
            llm_client=mock_llm,
        )

        from app.models.document import VisionPrediction

        mock_prediction = VisionPrediction(
            raw_class="Tomato___Early_blight",
            crop="tomato",
            disease="early blight",
            confidence=0.96,
            low_confidence=False,
            engine="hybrid",
        )

        with (
            unittest.mock.patch("app.services.rag_service.diagnose_image", return_value=mock_prediction),
            unittest.mock.patch("app.services.rag_service.retrieve", return_value=[make_chunk()]),
        ):
            response = service.handle_diagnose(
                image_bytes=b"fake-leaf-bytes",
                filename="leaf.jpg",
                content_type="image/jpeg",
                language="hi",
            )

            self.assertEqual(response.answer, "हिंदी में उपचार निर्देश [1].")
            called_prompt = mock_llm.generate.call_args[0][0]
            self.assertIn("LANGUAGE LOCALIZATION MANDATE (Hindi / हिन्दी)", called_prompt)


if __name__ == "__main__":
    unittest.main()

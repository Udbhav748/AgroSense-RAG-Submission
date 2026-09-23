"""Abstract interface for a language model client.

The rest of the application should depend on this interface, not on any
specific provider (e.g. Gemini), so the backing LLM can be swapped without
touching calling code.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from app.core.exceptions import LLMConfigurationError


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text completion for the given prompt."""

    def generate_structured(self, prompt: str) -> str:
        """Generate a JSON-mode completion for the given prompt.

        The prompt asks for a JSON object (see
        prompt_builder.build_structured_prompt); providers that support a
        JSON output mode (Gemini response_mime_type, Groq response_format)
        request it here. Default implementation: plain generate() — a
        client that doesn't implement JSON mode (or a test fake) just
        returns whatever text the model gives, and the caller's parse step
        decides whether it's usable.
        """
        return self.generate(prompt)

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Generate a text completion, yielding it in pieces as it's
        produced. Default implementation: no real streaming, just yields
        generate()'s full result once — this is what FallbackLLMClient
        inherits (streaming would mean re-streaming from the fallback
        provider mid-response if the primary failed partway, which isn't
        a sensible thing to do to a client that's already rendered
        partial output) and what any client/test that hasn't implemented
        real streaming falls back to. Concrete clients that support it
        (GeminiClient, GroqClient) override this with real streaming.
        """
        yield self.generate(prompt)

    def generate_with_image(
        self, prompt: str, image_bytes: bytes, mime_type: str = "image/png"
    ) -> str:
        """Generate a text completion from a prompt plus a single image —
        the multi-modal extension of generate() used by image captioning
        (image_captioning_service) and vision-grounded QA
        (vision_qa_service).

        Concrete clients that support image input (GeminiClient) override
        this; the default raises LLMConfigurationError so a client that
        can't see images — GroqClient, a test fake — fails loudly rather
        than silently captioning "nothing". Callers (image_captioning_
        service) treat this error as "this provider can't do images" and
        degrade to no caption chunk, exactly like a caption generation
        failure.
        """
        raise LLMConfigurationError(
            f"{type(self).__name__} does not support image input; "
            "image captioning / vision QA requires a vision-capable provider (Gemini)."
        )

"""Unit tests for Leaf Infection Explainability & Visual Saliency Heatmap Service."""

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.models.document import VisionPrediction
from app.models.schemas import DiagnosisInfo
from app.services.explainability import (
    _compute_edge_gradients,
    _count_connected_components,
    _rgb_to_hsv_numpy,
    _rgb_to_lab_numpy,
    generate_leaf_saliency,
)
from app.services.vision_client import diagnose_image


def _create_synthetic_leaf_image(
    width: int = 200,
    height: int = 200,
    has_lesions: bool = False,
    has_chlorosis: bool = False,
    severity: str = "mild",
) -> bytes:
    """Helper to generate synthetic leaf image bytes in memory."""
    # Green leaf background (RGB: [40, 160, 50])
    img = np.full((height, width, 3), [40, 160, 50], dtype=np.uint8)

    if has_chlorosis:
        # Yellow chlorosis margin (RGB: [220, 200, 30])
        img[50:150, 50:150] = [220, 200, 30]

    if has_lesions:
        if severity == "severe":
            # Multiple necrotic brown/black lesion spots (RGB: [50, 25, 15])
            img[60:110, 60:110] = [50, 25, 15]
            img[120:160, 120:160] = [30, 15, 10]
            img[30:70, 130:170] = [45, 20, 10]
        elif severity == "moderate":
            img[70:120, 70:120] = [50, 25, 15]
            img[130:155, 130:155] = [35, 20, 10]
        else:
            # Small mild lesion spot
            img[90:105, 90:105] = [50, 25, 15]

    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestExplainabilitySaliency:
    """Test suite for leaf saliency heatmap generation and segmentation."""

    def test_saliency_healthy_leaf(self):
        """Healthy green foliage should produce low infection percentage and Mild severity."""
        img_bytes = _create_synthetic_leaf_image(has_lesions=False, has_chlorosis=False)
        result = generate_leaf_saliency(img_bytes)

        assert isinstance(result, dict)
        assert result["heatmap_base64"] is not None
        assert isinstance(result["heatmap_base64"], str)
        assert len(result["heatmap_base64"]) > 50

        # Decode base64 to verify valid PNG structure
        decoded = base64.b64decode(result["heatmap_base64"])
        heatmap_img = Image.open(io.BytesIO(decoded))
        assert heatmap_img.format == "PNG"

        assert result["infected_area_percentage"] < 10.0
        assert result["severity_level"] == "Mild"
        assert result["lesion_count"] == 0

    def test_saliency_diseased_leaf_with_lesions(self):
        """Diseased leaf with necrotic lesions and chlorosis should detect infected areas and lesion count."""
        img_bytes = _create_synthetic_leaf_image(
            has_lesions=True, has_chlorosis=True, severity="severe"
        )
        result = generate_leaf_saliency(img_bytes)

        assert result["heatmap_base64"] is not None
        assert result["infected_area_percentage"] > 10.0
        assert result["lesion_count"] >= 1
        assert result["severity_level"] in ("Moderate", "Severe")

    def test_saliency_moderate_infection(self):
        """Moderate infection should compute accurate infected area percentage and severity level."""
        img_bytes = _create_synthetic_leaf_image(
            has_lesions=True, has_chlorosis=False, severity="moderate"
        )
        result = generate_leaf_saliency(img_bytes)

        assert result["heatmap_base64"] is not None
        assert result["infected_area_percentage"] > 0.0
        assert result["lesion_count"] >= 1
        assert result["severity_level"] in ("Mild", "Moderate", "Severe")

    def test_saliency_empty_bytes_fallback(self):
        """Empty bytes input should return graceful fallback dictionary without error."""
        result = generate_leaf_saliency(b"")

        assert result["heatmap_base64"] is None
        assert result["infected_area_percentage"] == 0.0
        assert result["lesion_count"] == 0
        assert result["severity_level"] == "Mild"

    def test_saliency_corrupted_bytes_fallback(self):
        """Corrupted/invalid image bytes should safely return fallback without throwing."""
        corrupted_bytes = b"NOT_A_VALID_IMAGE_DATA_CORRUPTED_STREAM_12345"
        result = generate_leaf_saliency(corrupted_bytes)

        assert result["heatmap_base64"] is None
        assert result["infected_area_percentage"] == 0.0
        assert result["lesion_count"] == 0
        assert result["severity_level"] == "Mild"

    def test_numpy_color_conversions(self):
        """Verify RGB to HSV, LAB, and edge gradient calculation utilities."""
        rgb_float = np.array([[[0.2, 0.8, 0.2], [0.9, 0.8, 0.1]]], dtype=np.float32)
        hsv = _rgb_to_hsv_numpy(rgb_float)
        assert hsv.shape == (1, 2, 3)
        assert hsv.dtype == np.uint8

        rgb_uint8 = np.array([[[50, 200, 50], [230, 200, 25]]], dtype=np.uint8)
        lab = _rgb_to_lab_numpy(rgb_uint8)
        assert lab.shape == (1, 2, 3)
        assert lab.dtype == np.uint8

        gray = np.zeros((50, 50), dtype=np.uint8)
        gray[20:30, 20:30] = 255
        edges = _compute_edge_gradients(gray)
        assert edges.shape == (50, 50)
        assert edges.dtype == np.uint8

    def test_count_connected_components(self):
        """Verify distinct lesion spot blob counting."""
        mask = np.zeros((100, 100), dtype=bool)
        # 2 distinct blobs > 10 pixels
        mask[10:20, 10:20] = True  # 100 px
        mask[60:75, 60:75] = True  # 225 px
        # 1 tiny noise spot < 5 pixels
        mask[90:92, 90:92] = True  # 4 px

        count = _count_connected_components(mask, min_pixel_size=10)
        assert count == 2


class TestExplainabilitySchemaIntegration:
    """Test suite verifying explainability fields integration across schemas and vision client."""

    def test_vision_prediction_schema_fields(self):
        """VisionPrediction should support optional heatmap_base64, infected_area_percentage, lesion_count."""
        pred = VisionPrediction(
            raw_class="Tomato___Early_blight",
            crop="tomato",
            disease="early blight",
            confidence=0.92,
            low_confidence=False,
            engine="leafsense",
            heatmap_base64="fake_base64_string",
            infected_area_percentage=18.5,
            lesion_count=4,
        )

        assert pred.heatmap_base64 == "fake_base64_string"
        assert pred.infected_area_percentage == 18.5
        assert pred.lesion_count == 4

        # Default values when omitted
        default_pred = VisionPrediction(
            raw_class="Tomato___healthy",
            crop="tomato",
            disease="healthy",
            confidence=0.95,
            low_confidence=False,
        )
        assert default_pred.heatmap_base64 is None
        assert default_pred.infected_area_percentage is None
        assert default_pred.lesion_count is None

    def test_diagnosis_info_schema_fields(self):
        """DiagnosisInfo schema should support explainability metrics."""
        diag = DiagnosisInfo(
            raw_class="Apple___Apple_scab",
            crop="apple",
            disease="apple scab",
            confidence=0.88,
            low_confidence=False,
            heatmap_base64="data:image/png;base64,abc123",
            infected_area_percentage=12.4,
            lesion_count=3,
        )

        assert diag.heatmap_base64 == "data:image/png;base64,abc123"
        assert diag.infected_area_percentage == 12.4
        assert diag.lesion_count == 3

    @patch("app.services.vision_client.httpx.post")
    def test_diagnose_image_attaches_saliency_data(self, mock_post):
        """diagnose_image should automatically run saliency explainability and attach to VisionPrediction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "class": "Tomato___Early_blight",
            "confidence": 0.94,
        }
        mock_post.return_value = mock_response

        img_bytes = _create_synthetic_leaf_image(has_lesions=True, has_chlorosis=True)
        prediction = diagnose_image(
            contents=img_bytes,
            filename="test_tomato_leaf.png",
            content_type="image/png",
            engine="leafsense",
        )

        assert isinstance(prediction, VisionPrediction)
        assert prediction.crop == "tomato"
        assert prediction.disease == "early blight"
        assert prediction.confidence == 0.94
        assert prediction.heatmap_base64 is not None
        assert prediction.infected_area_percentage is not None
        assert prediction.infected_area_percentage >= 0.0
        assert prediction.lesion_count is not None

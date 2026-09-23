"""Unit tests for monitoring/alert_webhook.py's best-effort webhook push."""

import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.alert_webhook import send_alert  # noqa: E402


class TestSendAlert:
    def test_no_webhook_url_is_a_noop(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = send_alert(None, "should never be sent")
        assert result is False
        mock_urlopen.assert_not_called()

    def test_empty_webhook_url_is_a_noop(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = send_alert("", "should never be sent")
        assert result is False
        mock_urlopen.assert_not_called()

    def test_successful_post_returns_true(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = send_alert("https://hooks.example.com/webhook", "alert text")
        assert result is True
        mock_urlopen.assert_called_once()

    def test_non_2xx_response_returns_false(self):
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.__enter__.return_value = mock_response
        with patch("urllib.request.urlopen", return_value=mock_response):
            result = send_alert("https://hooks.example.com/webhook", "alert text")
        assert result is False

    def test_network_failure_is_swallowed_not_raised(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = send_alert("https://hooks.example.com/webhook", "alert text")
        assert result is False

    def test_payload_is_json_with_text_field(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            send_alert("https://hooks.example.com/webhook", "hello world")

        sent_request = mock_urlopen.call_args[0][0]
        assert sent_request.data == b'{"text": "hello world"}'
        assert sent_request.get_header("Content-type") == "application/json"

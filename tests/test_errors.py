"""Error taxonomy: status mapping, server codes, retry semantics."""

from __future__ import annotations

import pytest

from kiomon import KiomonError, code_for_status, error_from_response, is_kiomon_error


def test_status_mapping_matches_the_worker_fallback_table() -> None:
    assert code_for_status(400) == "bad_request"
    assert code_for_status(401) == "unauthorized"
    assert code_for_status(402) == "quota_exceeded"
    assert code_for_status(403) == "forbidden"
    assert code_for_status(404) == "not_found"
    assert code_for_status(409) == "conflict"
    assert code_for_status(413) == "payload_too_large"
    assert code_for_status(429) == "rate_limited"
    assert code_for_status(500) == "internal_error"


def test_unmapped_statuses_fall_back_by_class() -> None:
    # A 504 from the edge is not this SDK's own timeout, which it raises separately.
    assert code_for_status(502) == "service_unavailable"
    assert code_for_status(503) == "service_unavailable"
    assert code_for_status(504) == "service_unavailable"
    assert code_for_status(418) == "bad_request"
    assert code_for_status(302) == "unknown"


def test_error_infers_a_code_from_status() -> None:
    assert KiomonError("boom", status=404).code == "not_found"
    assert KiomonError("boom").code == "unknown"
    assert KiomonError("boom", status=404).status == 404


def test_surfaces_the_api_message_and_marks_402_as_upgradeable() -> None:
    error = error_from_response(status=402, body=b'{"error": "Free plan limit reached", "needsUpgrade": true}')
    assert error.code == "quota_exceeded"
    assert error.needs_upgrade is True
    assert str(error) == "Free plan limit reached"
    assert error.retryable is False


def test_message_falls_back_to_the_message_key_then_to_status_text() -> None:
    assert str(error_from_response(status=400, body=b'{"message": "shape was wrong"}')) == "shape was wrong"
    assert str(error_from_response(status=500, fallback_message="Internal Server Error")) == "Internal Server Error"
    assert str(error_from_response(status=503)) == "Request failed with HTTP 503"


def test_api_error_key_wins_over_message_key() -> None:
    error = error_from_response(status=400, body=b'{"error": "primary", "message": "secondary"}')
    assert str(error) == "primary"


def test_a_server_supplied_code_wins_including_an_unknown_one() -> None:
    assert error_from_response(status=403, body=b'{"error": "nope", "code": "insufficient_role"}').code == (
        "insufficient_role"
    )
    assert error_from_response(status=400, body=b'{"error": "nope", "code": "brand_new"}').code == "brand_new"


def test_non_json_bodies_do_not_break_error_parsing() -> None:
    error = error_from_response(status=502, body=b"<html>bad gateway</html>")
    assert error.code == "service_unavailable"
    assert error.body is None


def test_request_id_and_retry_after_are_read_from_headers_case_insensitively() -> None:
    error = error_from_response(
        status=429,
        body=b'{"error": "slow"}',
        headers={"X-Request-Id": "req_1", "Retry-After": "3"},
    )
    assert error.request_id == "req_1"
    assert error.retry_after == 3.0


def test_body_retry_after_is_kept_when_larger() -> None:
    error = error_from_response(status=429, body=b'{"error": "slow", "retry_after": 5}', headers={"retry-after": "2"})
    assert error.retry_after == 5.0


def test_needs_upgrade_is_present_but_false_on_a_pro_rate_limit() -> None:
    error = error_from_response(status=429, body=b'{"error": "slow", "needsUpgrade": false}')
    assert error.needs_upgrade is False


@pytest.mark.parametrize(
    "code",
    ["rate_limited", "internal_error", "service_unavailable", "timeout", "network_error"],
)
def test_retryable_codes(code: str) -> None:
    assert KiomonError("boom", code=code).retryable is True


@pytest.mark.parametrize(
    "code",
    ["bad_request", "validation_failed", "unauthorized", "forbidden", "not_found", "quota_exceeded", "pro_required"],
)
def test_terminal_codes_are_not_retryable(code: str) -> None:
    assert KiomonError("boom", code=code).retryable is False


def test_is_kiomon_error_narrows() -> None:
    assert is_kiomon_error(KiomonError("boom")) is True
    assert is_kiomon_error(ValueError("boom")) is False


def test_repr_includes_code_and_status() -> None:
    text = repr(KiomonError("boom", status=429, code="rate_limited"))
    assert "rate_limited" in text
    assert "429" in text

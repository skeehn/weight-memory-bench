"""Tests for the remote driver's failure handling.

This code has produced three runtime bugs and had zero tests, which is the wrong ratio. All
three were in error handling -- the paths that only execute when something has already gone
wrong, and therefore the paths least likely to be exercised before they matter.

The retry classifier is the sharpest example: `HTTPError` subclasses `URLError`, so catching
`URLError` retried server errors while the docstring claimed it did not. The code and its
documentation disagreed, and only a live 4xx revealed which one was lying.
"""

from __future__ import annotations

import io
import os
import urllib.error

import pytest

os.environ.setdefault("WMB_BASETEN_DEPLOYMENT_ID", "test-deployment")

from scripts import run_benchmark as rb  # noqa: E402


def http_error(code: int, body: bytes = b"nope") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.invalid", code=code, msg="err", hdrs=None, fp=io.BytesIO(body)
    )


class TestRetryClassification:
    """4xx is a real answer about a real problem. 5xx and 429 are worth another try."""

    def _run(self, monkeypatch, errors, result=None):
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):
            i = calls["n"]
            calls["n"] += 1
            if i < len(errors):
                raise errors[i]

            class Response:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *a):
                    return False

                def read(self_inner):
                    import json

                    return json.dumps({"results": result or [{"id": "x", "text": "ok"}]}).encode()

            return Response()

        monkeypatch.setattr(rb.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(rb.time, "sleep", lambda *_: None)
        return calls

    def test_4xx_is_not_retried(self, monkeypatch):
        calls = self._run(monkeypatch, [http_error(400)])
        with pytest.raises(RuntimeError, match="not retryable"):
            rb.remote_generate([{"id": "x"}], "1B", "key")
        assert calls["n"] == 1, "a 400 must not be attempted more than once"

    def test_4xx_surfaces_the_response_body(self, monkeypatch):
        # Without the body, a 4xx is an opaque number and diagnosing it costs another run.
        self._run(monkeypatch, [http_error(422, b"payload too large")])
        with pytest.raises(RuntimeError, match="payload too large"):
            rb.remote_generate([{"id": "x"}], "1B", "key")

    def test_5xx_is_retried_then_succeeds(self, monkeypatch):
        calls = self._run(monkeypatch, [http_error(503), http_error(503)])
        out = rb.remote_generate([{"id": "x"}], "1B", "key")
        assert out == [{"id": "x", "text": "ok"}]
        assert calls["n"] == 3

    def test_429_is_retried(self, monkeypatch):
        calls = self._run(monkeypatch, [http_error(429)])
        rb.remote_generate([{"id": "x"}], "1B", "key")
        assert calls["n"] == 2

    def test_transport_errors_are_retried(self, monkeypatch):
        import ssl

        calls = self._run(
            monkeypatch, [ssl.SSLError("bad record mac"), urllib.error.URLError("broken pipe")]
        )
        rb.remote_generate([{"id": "x"}], "1B", "key")
        assert calls["n"] == 3

    def test_gives_up_after_the_attempt_budget(self, monkeypatch):
        calls = self._run(monkeypatch, [urllib.error.URLError("down")] * 10)
        with pytest.raises(RuntimeError, match="after 4 attempts"):
            rb.remote_generate([{"id": "x"}], "1B", "key", attempts=4)
        assert calls["n"] == 4

    def test_success_on_first_try_does_not_sleep(self, monkeypatch):
        slept = []
        monkeypatch.setattr(rb.time, "sleep", lambda s: slept.append(s))
        self._run(monkeypatch, [])
        monkeypatch.setattr(rb.time, "sleep", lambda s: slept.append(s))
        rb.remote_generate([{"id": "x"}], "1B", "key")
        assert slept == []


class TestEndpointRouting:
    def test_endpoint_targets_a_deployment_not_an_environment(self):
        # /environments/production/ routes to whatever was deployed previously, which has
        # already meant measuring the wrong build once.
        assert "/deployment/" in rb.ENDPOINT
        assert "/environments/" not in rb.ENDPOINT

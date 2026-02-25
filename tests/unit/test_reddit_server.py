"""Unit tests for reddit_ingestor.server — HTTP health/metrics server."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "reddit-ingestor"))

from reddit_ingestor.server import HealthServer


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class TestHealthServer:
    """Test HTTP handler logic directly (no network)."""

    def test_healthz_returns_200(self):
        from aiohttp.test_utils import make_mocked_request

        server = HealthServer(port=0)
        req = make_mocked_request("GET", "/healthz")
        resp = _run(server._healthz(req))
        assert resp.status == 200
        assert resp.text == "ok"

    def test_readyz_not_ready(self):
        from aiohttp.test_utils import make_mocked_request

        server = HealthServer(port=0)
        req = make_mocked_request("GET", "/readyz")
        resp = _run(server._readyz(req))
        assert resp.status == 503

    def test_readyz_after_set_ready(self):
        from aiohttp.test_utils import make_mocked_request

        server = HealthServer(port=0)
        server.set_ready(True)
        req = make_mocked_request("GET", "/readyz")
        resp = _run(server._readyz(req))
        assert resp.status == 200
        assert resp.text == "ready"

    def test_metrics_returns_prometheus_format(self):
        from aiohttp.test_utils import make_mocked_request

        server = HealthServer(port=0)
        req = make_mocked_request("GET", "/metrics")
        resp = _run(server._metrics(req))
        assert resp.status == 200
        assert b"reddit_ingestor" in resp.body

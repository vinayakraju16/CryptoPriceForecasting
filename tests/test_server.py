"""Tests for the dependency-free HTTP server (no Flask needed)."""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.client import HTTPConnection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crypto_prediction.server import ForecastService, make_handler  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

CSV_DIR = os.path.join(ROOT, "CSV")


class TestForecastService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(CSV_DIR):
            raise unittest.SkipTest("bundled CSVs missing")
        cls.service = ForecastService(CSV_DIR)

    def test_health(self):
        payload = self.service.health()
        self.assertEqual(payload["status"], "ok")
        self.assertGreaterEqual(payload["coins"], 1)

    def test_datasets_payload(self):
        payload = self.service.datasets_payload()
        self.assertEqual(payload["count"], len(payload["datasets"]))
        self.assertIn("Bitcoin", [row["symbol"] for row in payload["datasets"]])

    def test_forecast(self):
        payload = self.service.forecast({"coin": ["Bitcoin"], "model": ["drift"]})
        self.assertIn("predicted_next_price", payload)
        self.assertGreater(payload["last_price"], 0)

    def test_screen(self):
        payload = self.service.screen({"model": ["drift"], "top": ["3"]})
        self.assertLessEqual(payload["count"], 3)

    def test_recommend(self):
        payload = self.service.recommend({"model": ["drift"], "capital": ["1000"]})
        self.assertIn("proposals", payload)

    def test_backtest(self):
        payload = self.service.backtest({"coin": ["Bitcoin"], "model": ["drift"]})
        self.assertIn("metrics", payload)
        self.assertGreater(len(payload["predictions"]), 0)

    def test_stats(self):
        payload = self.service.stats({"coin": ["Dogecoin"]})
        self.assertEqual(payload["symbol"], "Dogecoin")

    def test_unknown_coin_raises(self):
        from crypto_prediction.core.dataset import DataError

        with self.assertRaises(DataError):
            self.service.forecast({"coin": ["NotACoin"]})


class TestHttpServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(CSV_DIR):
            raise unittest.SkipTest("bundled CSVs missing")
        service = ForecastService(CSV_DIR)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        conn.close()
        return response.status, body

    def test_healthz(self):
        status, body = self.get("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")

    def test_root_serves_html(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("Crypto Forecasting", body)

    def test_api_endpoints(self):
        for path in (
            "/api/datasets",
            "/api/models",
            "/api/forecast?coin=Bitcoin&model=drift",
            "/api/screen?model=drift&top=3",
            "/api/recommend?model=drift",
            "/api/backtest?coin=Bitcoin&model=drift",
            "/api/stats?coin=Bitcoin",
        ):
            status, body = self.get(path)
            self.assertEqual(status, 200, f"{path} -> {status} {body[:200]}")
            self.assertNotIn("Traceback", body)

    def test_unknown_route_is_json_404(self):
        status, body = self.get("/api/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(body))

    def test_error_responses_are_json_and_safe(self):
        status, body = self.get("/api/forecast?coin=NotACoin")
        self.assertEqual(status, 404)
        payload = json.loads(body)
        self.assertIn("error", payload)
        self.assertNotIn("/home/", body, "must not leak filesystem paths")

    def test_post_is_rejected(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request("POST", "/api/forecast", body="{}")
        self.assertEqual(conn.getresponse().status, 405)
        conn.close()

    def test_favicon_no_content(self):
        status, _ = self.get("/favicon.ico")
        self.assertEqual(status, 204)


if __name__ == "__main__":
    unittest.main(verbosity=2)

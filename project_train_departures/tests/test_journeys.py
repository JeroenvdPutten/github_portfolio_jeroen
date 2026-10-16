import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.ns import JourneyError, NSJourneyService, parse_response


NOW = datetime.now(timezone.utc)


def leg(origin="Haarlem", destination="Utrecht Centraal", minutes=10, number="100"):
    departure = NOW + timedelta(minutes=minutes)
    return {
        "origin": {"name": origin, "plannedDateTime": departure.isoformat(), "plannedTrack": "3"},
        "destination": {"name": destination, "plannedDateTime": (departure + timedelta(minutes=20)).isoformat()},
        "product": {"type": "TRAIN", "displayName": "NS Intercity", "number": number},
        "direction": destination,
        "cancelled": False,
    }


def trip(*legs, status="NORMAL"):
    return {"status": status, "legs": list(legs)}


class JourneyTests(unittest.TestCase):
    def service(self, payload=None, status=200, error=None):
        self.calls = []

        def respond(request):
            self.calls.append(request)
            if error:
                raise error
            return httpx.Response(status, json=payload)

        client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        return NSJourneyService(client, "test-key")

    def test_actual_times_tracks_and_transfer(self):
        first = leg(destination="Amsterdam Centraal")
        first["origin"]["actualDateTime"] = (NOW + timedelta(minutes=12)).isoformat()
        first["origin"]["actualTrack"] = "5b"
        second = leg("Amsterdam Centraal", minutes=35, number="200")
        result = parse_response({"trips": [trip(first, second)]}, "Haarlem", "Utrecht Centraal")[0]
        self.assertEqual(len(result.legs), 2)
        self.assertEqual(result.legs[0].departure_delay_minutes, 2)
        self.assertTrue(result.legs[0].departure_platform_changed)
        self.assertEqual(result.legs[0].departure_platform, "5b")
        self.assertIsNone(result.legs[0].arrival_platform)

    def test_cancelled_impossible_and_nontrain_options_excluded(self):
        cancelled = leg()
        cancelled["cancelled"] = True
        bus = leg()
        bus["product"]["type"] = "BUS"
        missed = trip(leg(destination="Amsterdam Centraal"), leg("Amsterdam Centraal", minutes=15))
        payload = {"trips": [trip(leg(), status="CANCELLED"), trip(cancelled), trip(bus),
                             trip(leg(), status="CHANGE_NOT_POSSIBLE"), missed, trip(leg())]}
        self.assertEqual(len(parse_response(payload, "Haarlem", "Utrecht Centraal")), 1)

    def test_three_distinct_future_departures_sorted(self):
        payload = {"trips": [trip(leg(minutes=m, number=str(m))) for m in [40, -1, 20, 10, 30, 10]]}
        service = self.service(payload)
        result = service.search("haarlem-utrecht")
        self.assertEqual([j.legs[0].train_number for j in result.journeys], ["10", "20", "30"])
        self.assertEqual(result.mode, "live")
        self.assertIsNotNone(result.updated_at)
        self.assertEqual(self.calls[0].url.params["fromStation"], "Haarlem")
        self.assertEqual(self.calls[0].headers["Ocp-Apim-Subscription-Key"], "test-key")

    def test_reverse_direction(self):
        service = self.service({"trips": [trip(leg("Utrecht Centraal", "Haarlem"))]})
        self.assertEqual(len(service.search("utrecht-haarlem").journeys), 1)
        self.assertEqual(self.calls[0].url.params["fromStation"], "Utrecht Centraal")

    def test_cache_reused_but_past_departures_removed(self):
        service = self.service({"trips": [trip(leg(minutes=10))]})
        self.assertEqual(len(service.search("haarlem-utrecht").journeys), 1)
        with patch("app.ns.datetime") as clock:
            clock.now.return_value = NOW + timedelta(minutes=11)
            self.assertEqual(service.search("haarlem-utrecht").journeys, [])
        self.assertEqual(len(self.calls), 1)

    def test_http_failures_are_safe_and_briefly_cached(self):
        for status in (401, 403, 429, 500):
            with self.subTest(status=status):
                service = self.service({"error": "private upstream body"}, status=status)
                for _ in range(2):
                    with self.assertRaises(JourneyError) as context:
                        service.search("haarlem-utrecht")
                    self.assertNotIn("private upstream body", str(context.exception))
                    self.assertNotIn("test-key", str(context.exception))
                self.assertEqual(len(self.calls), 1)

    def test_timeout_and_bad_json_shape(self):
        service = self.service(error=httpx.ReadTimeout("private failure"))
        with self.assertRaises(JourneyError) as context:
            service.search("haarlem-utrecht")
        self.assertEqual(context.exception.status_code, 504)
        with self.assertRaises(JourneyError):
            self.service({"unexpected": []}).search("haarlem-utrecht")

    def test_empty_and_malformed_journeys(self):
        malformed = leg()
        malformed["origin"]["plannedDateTime"] = "invalid"
        self.assertEqual(parse_response({"trips": []}, "Haarlem", "Utrecht Centraal"), [])
        payload = {"trips": [trip(malformed), trip(leg())]}
        self.assertEqual(len(parse_response(payload, "Haarlem", "Utrecht Centraal")), 1)

    def test_naive_datetime_rejected(self):
        malformed = leg()
        malformed["origin"]["plannedDateTime"] = "2030-01-01T10:00:00"
        self.assertEqual(parse_response({"trips": [trip(malformed)]}, "Haarlem", "Utrecht Centraal"), [])

    def test_web_routes_errors_and_demo(self):
        with TestClient(app) as browser:
            app.state.journeys = self.service({"trips": [trip(leg())]})
            response = browser.get("/api/journeys")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mode"], "live")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(browser.get("/api/journeys?direction=bad").status_code, 422)
            self.assertEqual(browser.get("/").status_code, 200)
            self.assertEqual(browser.get("/health").json(), {"status": "ok"})
            self.assertEqual(len(self.calls), 1)
            app.state.journeys = NSJourneyService(httpx.Client(), "")
            self.addCleanup(app.state.journeys.client.close)
            self.assertEqual(browser.get("/api/journeys").status_code, 503)
            with patch.dict(os.environ, {"APP_MODE": "demo"}):
                self.assertEqual(browser.get("/api/journeys").json()["mode"], "demo")


if __name__ == "__main__":
    unittest.main()

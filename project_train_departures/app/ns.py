"""Translate NS responses and share short-lived results between visitors."""

from datetime import datetime, timezone
from threading import Lock
from time import monotonic

import httpx

from app.schemas import Direction, Journey, Leg, SearchResult


TRIPS_URL = "https://gateway.apiportal.ns.nl/reisinformatie-api/api/v3/trips"
STATIONS = {
    "haarlem-utrecht": ("Haarlem", "Utrecht Centraal"),
    "utrecht-haarlem": ("Utrecht Centraal", "Haarlem"),
}
CACHE_SECONDS = 15


class JourneyError(Exception):
    """A safe message for the browser, without upstream credentials or bodies."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("NS timestamp has no timezone")
    return result


def parse_trip(trip: dict) -> Journey | None:
    """Keep complete, possible train journeys; never remove individual legs."""
    status = trip.get("status", "NORMAL")
    if status in {"CANCELLED", "CHANGE_NOT_POSSIBLE"} or trip.get("cancelled"):
        return None
    raw_legs = trip.get("legs", [])
    if not raw_legs:
        return None

    legs = []
    for raw in raw_legs:
        product = raw.get("product") or {}
        origin = raw.get("origin") or {}
        destination = raw.get("destination") or {}
        if (product.get("type") != "TRAIN" or raw.get("cancelled")
                or origin.get("cancelled") or destination.get("cancelled")):
            return None

        planned = origin.get("plannedDateTime")
        departure = timestamp(origin.get("actualDateTime") or planned)
        arrival = timestamp(destination.get("actualDateTime") or destination.get("plannedDateTime"))
        planned_departure = timestamp(planned) if planned else None
        if arrival < departure:
            return None
        if legs and (legs[-1].destination != origin["name"] or legs[-1].arrival > departure):
            return None

        planned_track = origin.get("plannedTrack")
        actual_track = origin.get("actualTrack")
        legs.append(Leg(
            origin=origin["name"],
            destination=destination["name"],
            departure=departure,
            arrival=arrival,
            train=product.get("displayName") or product.get("longCategoryName") or "Train",
            train_number=product.get("number"),
            train_destination=raw.get("direction") or destination["name"],
            departure_platform=actual_track or planned_track or None,
            arrival_platform=destination.get("actualTrack") or destination.get("plannedTrack") or None,
            planned_departure=planned_departure,
            departure_delay_minutes=max(0, round((departure - planned_departure).total_seconds() / 60)) if planned_departure else 0,
            departure_platform_changed=bool(actual_track and planned_track and actual_track != planned_track),
        ))

    warning = None
    if status != "NORMAL":
        warning = "NS reports a disruption or uncertainty for this journey. Check station information before travelling."
    return Journey(legs=legs, warning=warning)


def parse_response(data: dict, origin: str, destination: str) -> list[Journey]:
    if not isinstance(data, dict) or not isinstance(data.get("trips"), list):
        raise JourneyError("NS returned unexpected journey information. Please try again later.", 502)
    journeys = []
    for trip in data["trips"]:
        try:
            journey = parse_trip(trip)
        except (KeyError, TypeError, ValueError, AttributeError):
            # Malformed journeys must not prevent other complete options showing.
            continue
        if journey and journey.legs[0].origin == origin and journey.legs[-1].destination == destination:
            journeys.append(journey)
    return sorted(journeys, key=lambda item: (item.legs[0].departure, item.legs[-1].arrival))


class NSJourneyService:
    """One service instance owns the HTTP client and a cache for each direction.

    Locks prevent concurrent searches from making duplicate NS calls. Cache
    failures briefly too, so an NS outage does not cause repeated requests.
    """

    def __init__(self, client: httpx.Client, api_key: str):
        self.client = client
        self.api_key = api_key
        self.locks = {direction: Lock() for direction in STATIONS}
        self.cache = {}
        self.failures = {}

    def search(self, direction: Direction) -> SearchResult:
        searched_at = datetime.now(timezone.utc)
        if not self.api_key:
            raise JourneyError("Live train information is not configured yet. Please contact the app owner.")
        with self.locks[direction]:
            now = monotonic()
            failure = self.failures.get(direction)
            if failure and now < failure[0]:
                raise failure[1]
            cached = self.cache.get(direction)
            if cached is None or now >= cached[0]:
                try:
                    journeys = self.fetch(direction, searched_at)
                except JourneyError as error:
                    self.failures[direction] = (monotonic() + CACHE_SECONDS, error)
                    raise
                cached = (monotonic() + CACHE_SECONDS, datetime.now(timezone.utc), journeys)
                self.cache[direction] = cached
                self.failures.pop(direction, None)

            upcoming = []
            seen = set()
            for journey in cached[2]:
                first = journey.legs[0]
                if first.departure < searched_at:
                    continue
                # Different onward connections can share the same first train.
                identity = (first.train_number or first.train_destination, first.departure)
                if identity in seen:
                    continue
                seen.add(identity)
                upcoming.append(journey)
                if len(upcoming) == 3:
                    break
            return SearchResult(mode="live", searched_at=searched_at, updated_at=cached[1], journeys=upcoming)

    def fetch(self, direction: Direction, searched_at: datetime) -> list[Journey]:
        origin, destination = STATIONS[direction]
        try:
            response = self.client.get(TRIPS_URL, params={
                "fromStation": origin,
                "toStation": destination,
                "dateTime": searched_at.isoformat(timespec="seconds"),
                "searchForArrival": "false",
                # A delayed train planned in the past may still be upcoming.
                "previousAdvices": 2,
                "nextAdvices": 6,
            }, headers={"Ocp-Apim-Subscription-Key": self.api_key})
        except httpx.TimeoutException:
            raise JourneyError("NS took too long to respond. Please try again shortly.", 504) from None
        except httpx.RequestError:
            raise JourneyError("Cannot reach NS right now. Please try again shortly.") from None
        if response.status_code in {401, 403}:
            raise JourneyError("NS access is unavailable. The app owner needs to check the API subscription.")
        if response.status_code == 429:
            raise JourneyError("NS is receiving too many requests. Please wait a moment and retry.")
        if response.status_code != 200:
            raise JourneyError("NS journey information is temporarily unavailable. Please try again shortly.", 502)
        try:
            data = response.json()
        except ValueError:
            raise JourneyError("NS returned unreadable journey information. Please try again later.", 502) from None
        return parse_response(data, origin, destination)

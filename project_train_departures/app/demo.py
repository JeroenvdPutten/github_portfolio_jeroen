"""Invented journeys for learning. These are never real travel advice."""

from datetime import datetime, timedelta, timezone

from app.schemas import Direction, Journey, Leg, SearchResult


def search_demo(direction: Direction) -> SearchResult:
    # Aware UTC timestamps avoid depending on the container's local timezone.
    now = datetime.now(timezone.utc)
    origin, destination = "Haarlem", "Utrecht Centraal"
    if direction == "utrecht-haarlem":
        origin, destination = destination, origin

    journeys = []
    for minutes in (5, 20, 35):
        departure = now + timedelta(minutes=minutes)
        # Include a transfer so the UI can be built before NS access is ready.
        legs = [
            Leg(
                origin=origin,
                destination="Amsterdam Sloterdijk",
                departure=departure,
                arrival=departure + timedelta(minutes=15),
                train="Example Intercity",
                train_destination="Amsterdam Centraal",
                departure_platform="3",
                arrival_platform="5",
            ),
            Leg(
                origin="Amsterdam Sloterdijk",
                destination=destination,
                departure=departure + timedelta(minutes=23),
                arrival=departure + timedelta(minutes=55),
                train="Example Intercity",
                train_destination=destination,
                departure_platform="7",
                arrival_platform=None,
            ),
        ]
        journeys.append(Journey(legs=legs))
    return SearchResult(mode="demo", searched_at=now, journeys=journeys)

r"""Make one NS request without changing the browser app's demo mode.

Run from this directory: .\.venv\Scripts\python.exe check_ns.py
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv


URL = "https://gateway.apiportal.ns.nl/reisinformatie-api/api/v3/trips"


def main() -> int:
    # An existing environment variable takes priority over the local file.
    load_dotenv(Path(__file__).parent / ".env")
    key = os.environ.get("NS_API_KEY", "").strip()
    if not key or key == "your_actual_key_here":
        print("Set NS_API_KEY in this project's .env file first.")
        return 1

    params = {
        "fromStation": "Haarlem",
        "toStation": "Utrecht Centraal",
        "dateTime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "searchForArrival": "false",
    }
    headers = {"Ocp-Apim-Subscription-Key": key}

    print("Requesting Haarlem -> Utrecht Centraal from NS...")
    try:
        response = httpx.get(URL, params=params, headers=headers, timeout=20)
    except httpx.RequestError:
        print("Could not reach NS. Check your connection and try again.")
        return 1

    print(f"HTTP status: {response.status_code}")
    if response.status_code != 200:
        explanations = {
            400: "NS rejected the parameters. Check the operation's portal documentation.",
            401: "Check the key and whether your subscription is active.",
            403: "Check whether your subscription grants access to Reisinformatie API.",
            429: "The request limit was reached. Wait before retrying.",
        }
        print(explanations.get(response.status_code, "NS did not return a successful response."))
        # Avoid printing request headers or raw error bodies containing secrets.
        return 1

    try:
        data = response.json()
    except ValueError:
        print("NS returned a response that is not valid JSON.")
        return 1

    if not isinstance(data, dict) or not isinstance(data.get("trips"), list):
        print("Unexpected response shape: expected an object containing a trips list.")
        return 1

    trips = data["trips"]
    print(f"Top-level JSON fields: {', '.join(data.keys())}")
    print(f"Journeys returned: {len(trips)}")
    print("Showing the first three returned options; this probe does not filter journeys yet.")
    for index, trip in enumerate(trips[:3], start=1):
        print(f"\nJourney {index}: status={trip.get('status', 'unknown')}")
        for leg in trip.get("legs", []):
            origin = leg.get("origin") or {}
            destination = leg.get("destination") or {}
            product = leg.get("product") or {}
            departure = origin.get("actualDateTime") or origin.get("plannedDateTime")
            platform = origin.get("actualTrack") or origin.get("plannedTrack") or "unknown"
            train = product.get("displayName") or product.get("longCategoryName") or "unknown"
            print(f"  {origin.get('name', '?')} -> {destination.get('name', '?')}")
            print(f"  Departure: {departure}; platform: {platform}; train: {train}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

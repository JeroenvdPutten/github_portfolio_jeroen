"""Classes describing the JSON our browser receives."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Direction = Literal["haarlem-utrecht", "utrecht-haarlem"]


class Leg(BaseModel):
    origin: str
    destination: str
    departure: datetime
    arrival: datetime
    train: str
    train_destination: str
    departure_platform: str | None
    arrival_platform: str | None
    train_number: str | None = None
    planned_departure: datetime | None = None
    departure_delay_minutes: int = 0
    departure_platform_changed: bool = False


class Journey(BaseModel):
    legs: list[Leg] = Field(min_length=1)
    warning: str | None = None


class SearchResult(BaseModel):
    mode: Literal["demo", "live"] = "live"
    searched_at: datetime
    updated_at: datetime | None = None
    journeys: list[Journey]

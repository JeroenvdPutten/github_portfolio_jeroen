"""HTTP routes: serve the browser page and respond to journey searches."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse

from app.demo import search_demo
from app.ns import JourneyError, NSJourneyService
from app.schemas import Direction, SearchResult


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One client reuses connections and one service shares cached results.
    with httpx.Client(timeout=12) as client:
        app.state.journeys = NSJourneyService(client, os.environ.get("NS_API_KEY", "").strip())
        yield


app = FastAPI(title="Haarlem / Utrecht trains", version="0.2.0", lifespan=lifespan)
PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PAGE)


@app.get("/api/journeys", response_model=SearchResult)
def journeys(request: Request, response: Response, direction: Direction = "haarlem-utrecht"):
    response.headers["Cache-Control"] = "no-store"
    if os.environ.get("APP_MODE", "live") == "demo":
        return search_demo(direction)
    try:
        return request.app.state.journeys.search(direction)
    except JourneyError as error:
        raise HTTPException(error.status_code, detail=str(error), headers={"Cache-Control": "no-store", "Retry-After": "15"}) from None


@app.get("/health", include_in_schema=False)
def health():
    # Hosting health checks must not consume NS API calls.
    return {"status": "ok"}

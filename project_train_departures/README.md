# Haarlem ↔ Utrecht trains

A Python/FastAPI browser app showing up to three upcoming train departures
between Haarlem and Utrecht Centraal, in either direction. It uses NS journey
planning for transfers and displays train numbers, destinations, departure and
arrival platforms, departure delays, and departure platform changes.

Cancelled journeys, impossible transfers, and options requiring non-train legs
are excluded. Fewer than three results can be returned if NS supplies fewer
usable options. Times are displayed in Europe/Amsterdam. Always check station
displays before boarding.

## Run locally first (PowerShell, Python 3.12+)

From the repository root:

```powershell
cd project_train_departures
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `.env` in this directory using `.env.example` as a guide:

```dotenv
NS_API_KEY=your_actual_key_here
APP_MODE=live
```

Keep the real key local. `.env` is excluded from Git and the Docker build.
Then start the app:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open http://localhost:8000 and search. Stop with Ctrl+C.
The API playground is at http://localhost:8000/docs.

To deliberately use invented journeys without NS access, set `APP_MODE=demo`
in `.env` and restart. Results are labelled DEMO. Failed live requests never
silently fall back to sample data.

## Then run the same app with Docker

Start Docker Desktop with Linux containers enabled. Stop any local Python
server using port 8000. From this directory:

```powershell
docker compose up --build
```

Open http://localhost:8000. Stop with Ctrl+C, then `docker compose down`.
Compose's `env_file` passes `.env` values into the container at runtime; the file
is not copied into the image. Rebuild after changing code.

The container starts `python -m app.start`, which runs Uvicorn on `0.0.0.0`
using the `PORT` environment variable (default 8000). Compose fixes that value
to 8000 for the local port mapping.

## Put it on the internet with Render

Hosting runs the Docker container on a remote server. Render supplies a public
HTTPS address and handles incoming traffic. Your laptop can then be switched off.
The browser calls our Python server; only the server calls NS with the secret key.

This repository includes `render.yaml` for a free Render web service. No hosting
account is created automatically, and local changes are not automatically pushed.

1. Create an account at https://dashboard.render.com/ (GitHub sign-in is useful).
2. Commit and push this project's source files to your GitHub repository. Do not
   include `.env`, `.venv`, or a subscription key. Confirm the files appear online.
3. In Render choose **New → Web Service**, connect GitHub, and select the repository.
   Grant repository access only where needed.
4. Use these settings:

   | Setting | Value |
   | --- | --- |
   | Name | A unique name, such as `haarlem-utrecht-trains` |
   | Root Directory | `project_train_departures` |
   | Language / Runtime | Docker |
   | Dockerfile Path | `./Dockerfile` |
   | Docker Build Context | `.` |
   | Region | Frankfurt, if available |
   | Instance Type | Free |
   | Health Check Path | `/health` |

5. Under **Environment**, add `NS_API_KEY` with your real key, and `APP_MODE`
   with value `live`. Leave the Docker command override empty to use the image's
   command. Never put the key into a source file or Docker build argument.
6. Create the web service and follow its deploy logs. Once live, open the HTTPS
   URL Render provides. Search both directions and check times and platforms.

Alternatively, create a **Blueprint** from this repository and set its Blueprint
Path to `project_train_departures/render.yaml`. It will prompt for `NS_API_KEY`.

Free Render services sleep after 15 minutes without traffic and can take about a
minute to wake up. Free services also have monthly usage limits; this is suitable
for learning and occasional personal use. A paid service is an option later if
you need it always running. Read the displayed plan details before creating it.

Official instructions: https://render.com/docs/web-services,
https://render.com/docs/docker, https://render.com/docs/free.

## How live searches work

`index.html → GET /api/journeys → NSJourneyService → NS /api/v3/trips → schemas → browser`

The service requests a small window of options around the current time, converts
NS data to our models, sorts departures, removes departures before the search
instant, and selects up to three distinct first trains. NS calculates transfers.
Actual times and platforms take priority over planned values when available.

Each direction shares a 15-second server cache. Cached results are still filtered
against the new search time. Failures are briefly cached too, to reduce repeated
calls during outages. This cache is in memory, per server process; use one Uvicorn
worker for this learning deployment. The response shows when NS data was fetched.

Timeouts, subscription problems, NS errors, missing configuration, and no results
have separate handling. The browser receives safe messages without request
headers, API keys, or upstream error bodies. `/health` does not call NS.

## Check the application

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Tests use simulated NS responses and make no external API calls.
To inspect real data separately:

```powershell
.\.venv\Scripts\python.exe check_ns.py
```

See LEARNING.md for explanations and exercises.

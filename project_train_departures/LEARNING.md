# Learn by following one search

## 1. Trace the request

Run the app locally before learning Docker. In the browser's developer tools,
open Network and press **Find next trains**. Find `/api/journeys`.

Follow this path:

`index.html submit handler → HTTP GET → main.py journeys() → demo.py search_demo() → JSON → browser cards`

Open `/docs` and call the endpoint yourself. Try an invalid direction. FastAPI
returns HTTP 422 because `Direction` allows only two literal strings.

Exercise: explain why the browser calls `/api/journeys` but never imports Python.

## 2. Your first classes: schemas.py

A dictionary can hold arbitrary keys. `Leg` describes a specific shape and types.
It inherits validation and serialization behavior from Pydantic's `BaseModel`.
You do not need to write a custom constructor for this model.

In `demo.py`, `Leg(origin=..., ...)` constructs an **instance** of the `Leg`
class. Each instance holds one train leg's data. `Journey` groups those instances;
`SearchResult` groups journeys with their search timestamp.

In an interactive Python session from this project directory, try:

```python
from app.schemas import Journey

journey = Journey(legs=[])
print(journey.legs)          # Attribute access, instead of dictionary indexing
print(journey.model_dump()) # Turn the model into a dictionary
Journey(legs="wrong")      # Observe the validation error
```

Why does our real provider need additional rules beyond type validation? For
example, an empty journey has the correct type but cannot be displayed as a trip.

Exercise: add a `train_number: str | None = None` field to `Leg`, supply an example
in `demo.py`, and display it beside the train type in the browser.

## 3. __init__.py versus __init__

`app/__init__.py` marks `app` as a regular Python package. It can be empty. Its
top-level code runs when Python first imports the package in a process, so keep
it simple. `from app.schemas import Leg` imports a name from a module in it.

`__init__` is a different thing: an instance initialization method on a class.
Python calls it during ordinary instance construction. Pydantic supplies the
initialization behavior for our models. A plain learning example:

```python
class Station:
    def __init__(self, name):
        self.name = name

station = Station("Haarlem")
print(station.name)
```

`self` refers to the instance being initialized. We do not need a `Station`
class in this app yet; plain station strings are sufficient.

## 4. Docker, one line at a time

Read the Dockerfile alongside these explanations:

- `FROM`: start with an image containing Python and a small Linux environment.
- `WORKDIR`: choose the directory for subsequent commands inside the image.
- `ENV`: configure Python output and bytecode behavior.
- `COPY requirements.txt` and `RUN pip install`: install dependencies at build
  time. Docker can reuse this layer when only application code changes.
- `COPY app`: include the source code in the image.
- `USER`: run the application as an unprivileged user.
- `EXPOSE`: document the container's intended port; it does not publish it.
- `CMD`: run `app.start`, which starts Uvicorn using the host's `PORT` setting.

An **image** is the packaged template. A **container** is a running instance of
that image. Compose describes how to build and run it. Its port mapping forwards
your computer's localhost port 8000 to port 8000 inside the container. Uvicorn
listens on `0.0.0.0` inside the container so that forwarding can reach it.

Exercise: change the host port to `127.0.0.1:8001:8000`. Which browser URL changes?
Does the Python code change? Edit the page and rebuild: why is rebuilding needed?

## 5. After NS access

Run `check_ns.py` using the README commands. Read its `main()` function:

- `load_dotenv` loads local file values into the process environment. Python
  and Docker do not automatically load a `.env` file into application code.
- `os.environ.get` reads the key without hardcoding it in source.
- `params` becomes the URL query string: which stations and when to depart.
- `headers` carries the subscription key, identifying our subscription to NS.
- `httpx.get` sends an HTTP GET request, with a 20-second timeout.
- `response.status_code` reports the HTTP outcome; `response.json()` turns
  JSON into Python dictionaries and lists.
- The `if __name__ == "__main__"` guard runs `main()` when the file is executed
  directly, but avoids making a request merely because another file imports it.

Exercise: explain why the key belongs in `headers` while stations belong in
`params`. Find the fallback from actual departure time/platform to planned values.
What should the browser show if neither platform field exists?

In `app/ns.py`, trace one real response. Identify journeys,
legs, planned versus actual times, platforms and cancellation fields. Keep HTTP
calls in a separate provider module; keep routes small and schemas independent
of NS's response structure. We will add tests for response mapping and failures.

Next learn timeouts, status codes, secrets supplied at runtime, and aware
timestamps. The demo uses UTC timestamps; the browser explicitly formats them in
Europe/Amsterdam, including daylight-saving changes.

For enterprise use we would later add locked dependencies, automated checks,
image scanning, operational logging, health checks, and managed secrets. This
app remains a learning project suitable for a small personal deployment.

## 6. Classes that own state: NSJourneyService

Read `app/ns.py`. Unlike a data model, this class has behavior and owns resources.
`__init__` receives the HTTP client and key, then initializes caches and locks.
`self.client` belongs to this instance. `search()` uses those saved attributes on
each call. The FastAPI lifespan in `main.py` creates one service instance when
the server starts and closes its HTTP client when the server stops.

The mapping functions stay outside the class because they only transform data;
they do not need to remember previous searches. This is a useful distinction
when choosing between a function and a class.

Exercise: why create one service for the server rather than constructing one
inside every route call? What would happen to the cache?

## 7. Hosting and runtime configuration

Localhost refers to the computer opening the URL. A remote host runs our image
and gives it an internet address. HTTPS requests reach the host, which forwards
them to Uvicorn inside the container. The page and API use the same origin, so
the browser can continue calling `/api/journeys` after deployment.

Locally, Compose's `env_file` supplies the key to the container. In Render, set
`NS_API_KEY` in the service's Environment settings instead. Neither approach
requires baking the key into the image. Do not print or return it in responses.
`PORT` is another runtime setting: `app/start.py` reads the host-assigned port.

Exercise: why does uploading just `index.html` fail to provide live NS searches?
Which process needs the key? Why does `/health` avoid calling NS?

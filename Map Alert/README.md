# CityPulse

CityPulse is a smart city incident dashboard for Tunis. It brings reports from future City Brain agents into one live map.

## What is included

- React, TypeScript, Vite and Leaflet map centered on Tunis.
- OpenStreetMap base map.
- FastAPI incident API with validation and an OpenAPI page.
- PostgreSQL with PostGIS geography storage.
- Live incident updates through WebSockets.
- Filters for domain, severity and status, plus incident details and a per-domain heatmap view.
- Simulated incidents for security, mobility, energy, environment, emergency and disaster.
- Agent integration guide in `AGENT_INTEGRATION.md`.

## Quick start with Docker Desktop

### 1. Start Docker Desktop

Open Docker Desktop and wait for the engine to finish starting. Keep Docker Desktop open while using CityPulse.

### 2. Open a terminal in this folder

The terminal must be in the folder containing `docker-compose.yml`. In Command Prompt, use:

```cmd
cd /d "C:\Users\azizb\Downloads\Map Alert"
```

In PowerShell, use:

```powershell
cd "C:\Users\azizb\Downloads\Map Alert"
```

If you moved this folder, replace the path with its new location.

### 3. Build and start the app

```text
docker compose up --build
```

The first build downloads the runtime images and packages and may take several minutes. Later builds usually reuse Docker's cache. Wait until the logs show the API running and the database healthy.

### 4. Open the right address

- **CityPulse dashboard:** http://localhost:5173
- **API interactive docs:** http://localhost:8000/docs
- **API health check:** http://localhost:8000/health

Do not browse to `0.0.0.0`. That address tells the server to listen on network interfaces; it is not the browser address.

### 5. Load example alerts

On the dashboard, click **Load demo data**. This adds ten simulated incidents in and around central Tunis. Use the domain buttons, severity selector, status selector, and search box to filter them. Click a map marker or incident card to see details. Turn on **Heatmap** to show incident intensity as larger translucent circles.

## Stop, restart and troubleshoot

- Stop the foreground stack: press `Ctrl+C` in the terminal running Compose.
- Start again without rebuilding: run `docker compose up` from this folder.
- Run in the background: `docker compose up --build -d`.
- Follow logs: `docker compose logs -f`.
- Stop background containers: `docker compose down`.
- `docker compose down` keeps the database volume and its incident data.
- To erase the local database and all saved incidents: `docker compose down -v`.

Common problems:

- **Docker API / named pipe error:** open Docker Desktop and wait for its engine to start, then retry.
- **No configuration file provided:** change to the `Map Alert` folder first; it contains `docker-compose.yml`.
- **Browser says it cannot reach `0.0.0.0:8000`:** use http://localhost:5173 for the dashboard or http://localhost:8000/docs for API docs.
- **Port is already allocated:** close the other app using port 5173, 8000, or 5432, or stop an older CityPulse stack with `docker compose down`.
- **Map tiles are blank:** the OpenStreetMap tiles need browser internet access.

## Run tests

With the Docker stack running, use this from the project folder:

```text
docker compose exec api pytest -q
```

The test suite covers API health, incident creation, filtering, duplicate IDs, validation, status updates, demo seeding and heatmap data.

## Agent API quick reference

Base URL: `http://localhost:8000`

- `GET /api/v1/incidents`
- `GET /api/v1/incidents/{id}`
- `POST /api/v1/incidents`
- `PATCH /api/v1/incidents/{id}`
- `GET /api/v1/heatmap?domain=environment`
- `POST /api/v1/demo/seed`
- WebSocket: `ws://localhost:8000/api/v1/ws`

Incident domains: `security`, `mobility`, `energy`, `environment`, `emergency`, `disaster`.

Severity values: `low`, `medium`, `high`, `critical`.

Status values: `active`, `investigating`, `resolved`.

Example incident body:

```json
{
  "title": "Traffic slowdown on Avenue Habib Bourguiba",
  "description": "Average speed below 10 km/h",
  "domain": "mobility",
  "severity": "high",
  "status": "active",
  "latitude": 36.8008,
  "longitude": 10.18,
  "source": "mobility-agent",
  "external_id": "camera-17-event-0001",
  "metadata": { "road": "Avenue Habib Bourguiba", "confidence": 0.92 }
}
```

The required fields are `title`, `domain`, `severity`, `latitude`, and `longitude`. Include a stable `external_id` so retrying an alert does not create duplicates; duplicate values return HTTP 409. Coordinates are WGS84 latitude and longitude. Put agent-specific fields in `metadata`. See `AGENT_INTEGRATION.md` for more examples and event details.

## Push this folder to GitHub

This folder is ready to upload as a repository. It includes `.gitignore`, which excludes local environments, database files, caches, and built frontend output.

### GitHub website or GitHub Desktop

1. Create a new empty repository on GitHub. Do not add a second README or license if you want to upload this README unchanged.
2. In GitHub Desktop, choose **Add an Existing Repository** and select the `Map Alert` folder. If it is not already a Git repository, GitHub Desktop will offer to create one.
3. Review the changed files, enter a message such as `Add CityPulse MVP`, commit, then choose **Publish repository**.

### Git command line

Create an empty repository on GitHub first, then replace `YOUR-USERNAME` below with your GitHub username and run these commands inside `Map Alert`:

```text
git init
git add .
git commit -m "Add CityPulse MVP"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Map-Alert.git
git push -u origin main
```

If Git says a remote named `origin` already exists, inspect it with `git remote -v` instead of adding it again. GitHub may ask you to sign in through a browser or credential manager.

## Configuration and current limitations

- Compose exposes the dashboard on port 5173, API on 8000, and PostgreSQL on 5432.
- The local Compose database credentials are for development only. Do not reuse them for a public deployment.
- This MVP does not include login or agent API authentication. Keep it on a trusted network until authentication is added.
- Heatmap data is returned as weighted points and drawn as circles; it is not a raster GIS heatmap yet.
- OpenStreetMap tiles require internet access and must be used in line with the OSM tile usage policy.

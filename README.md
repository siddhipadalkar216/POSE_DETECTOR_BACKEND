# Pose Coach API — Backend (Flask)

The backend manages accounts, exercise metadata, profiles, workout history, and speech synthesis. Real-time pose detection and coaching run in the browser using MediaPipe Tasks; video frames are not sent to the backend. This keeps the API deployment small enough for Vercel Functions.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:APP_ENV = 'development'
$env:DATABASE_URL = 'sqlite:///dev.db'
python -m app
```

The lightweight production dependencies are in `requirements.txt`. The old server-side analyzer is kept for local experiments only; install `requirements-pose-analyzer.txt` if you specifically need that module. It is not part of the deployed API.

## Environment variables

- `APP_ENV=production` enables production checks.
- `FLASK_SECRET_KEY` must be a random secret of at least 32 characters in production.
- `DATABASE_URL` should point to persistent PostgreSQL in production. Use an externally reachable database URL when the API runs on Vercel.
- `CORS_ORIGINS` is a comma-separated list of exact frontend origins, without trailing slashes.
- `SARVAM_API_KEY` is optional and enables generated speech; keep it on the backend host only.
- `SARVAM_TTS_MODEL` defaults to `bulbul:v3`.
- `TTS_API_URL`, `TTS_HEALTH_URL`, `TTS_API_KEY`, and `TTS_MODEL` are optional for a self-hosted compatible speech service.

## API

- `GET /api/v1/health`
- `GET /api/v1/tts/status`
- `POST /api/v1/tts/speech`
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET` and `PUT /api/v1/profile`
- `GET /api/v1/exercises`
- `POST /api/v1/logs`
- `GET /api/v1/logs`

`POST /api/v1/pose/analyze` now returns `410 Gone` for older clients. Current clients analyze camera frames locally and send only workout summaries to the API.

## Deploying the API to Vercel

Set the Vercel project root to the directory containing `app.py` and `requirements.txt`. Keep the Flask preset and let Vercel install `requirements.txt`. Configure `APP_ENV`, `FLASK_SECRET_KEY`, `DATABASE_URL`, and `CORS_ORIGINS` in the Vercel project environment. Vercel's Flask entry point is the module-level `app` in `app.py`.

For the Render website + Vercel API setup, see [`../DEPLOYMENT.md`](../DEPLOYMENT.md).

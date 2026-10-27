# Pose Coach API — Backend (Flask)

This is the Flask backend for the Exercise Pose Detector app.

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```
pip install -r requirements.txt
```

Optional (MMPose backend instead of MediaPipe):

```
# Install PyTorch + MMEngine + MMDetection/MMPose per their docs (versions depend on your CUDA)
# Example (CPU only):
pip install mmpose mmengine opencv-python
```

3. Environment variables (create `.env` at repo root):

```
FLASK_SECRET_KEY=
DATABASE_URL=sqlite:///dev.db
APP_ENV=development
AUTH_TOKEN_TTL_SECONDS=2592000
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Optional self-hosted, OpenAI-compatible TTS service
TTS_API_URL=
TTS_HEALTH_URL=
TTS_API_KEY=
TTS_MODEL=svara-tts-v1

# Recommended managed AI voice provider (kept server-side)
SARVAM_API_KEY=
SARVAM_TTS_MODEL=bulbul:v3

# Optional Azure
AZURE_ENDPOINT=
AZURE_API_KEY=
AZURE_RESOURCE_NAME=

# Optional MMPose
USE_MMPOSE=0
MEDIAPIPE_MODEL_COMPLEXITY=1  # 0 is faster, 2 uses more compute for fuller landmarks
MMPOSE_CONFIG=   # e.g., rtmo/rtmo-l_8xb16-700e_aic-coco-256x192.py
MMPOSE_CHECKPOINT=
MMPOSE_DEVICE=cpu  # or cuda:0
```

4. Run the server:

```
python -m server.app
```

## Database

- Default: SQLite via `DATABASE_URL=sqlite:///dev.db` (auto-created).
- Production: set `DATABASE_URL` to your PostgreSQL URL, e.g. `postgresql+psycopg2://user:pass@host:5432/dbname`.
- Models: `Exercise`, `AttemptLog`, optional `User`.

## API

- `GET /api/v1/health`
- `GET /api/v1/tts/status`
- `POST /api/v1/tts/speech` (authenticated; proxies audio to the configured TTS service)
- `GET /api/v1/exercises`
- `POST /api/v1/pose/analyze`
- `POST /api/v1/logs`
- `GET /api/v1/logs`

### Example curl

Analyze from a JPEG file (PowerShell):

```
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes('sample.jpg'))
Invoke-RestMethod -Method Post -Uri http://localhost:5001/api/v1/pose/analyze -ContentType 'application/json' -Body (ConvertTo-Json @{ image_base64 = $b64 })
```

Form-data alternative:

```
curl -X POST http://localhost:5001/api/v1/pose/analyze -F frame=@sample.jpg
```

Log an attempt:

```
curl -X POST http://localhost:5001/api/v1/logs \
  -H "Content-Type: application/json" \
  -d '{"exercise_id":1,"score":0.9,"suggestions":["Keep torso upright"],"landmarks":[]}'
```

Fetch logs:

```
curl http://localhost:5001/api/v1/logs
```

Production deployment to Render is documented in [`../DEPLOYMENT.md`](../DEPLOYMENT.md). Set Render's Root Directory to `server`, Python version to `3.11.11`, and use Gunicorn with `app:create_app()`. Production requires a strong `FLASK_SECRET_KEY`, a persistent PostgreSQL `DATABASE_URL`, and the deployed client's exact origin in `CORS_ORIGINS`.

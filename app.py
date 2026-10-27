import os
import base64
import io
import re
from datetime import datetime

from flask import Flask, Response, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import requests

# Support running both as a package (python -m server.app) and as a script (python server/app.py)
try:
    from .models import db_session, init_engine_and_session, Exercise, AttemptLog, User
    from .pose_analyzer import PoseAnalyzer
    from . import azure_client
except Exception:  # fallback when __package__ is None
    from models import db_session, init_engine_and_session, Exercise, AttemptLog, User  # type: ignore
    from pose_analyzer import PoseAnalyzer  # type: ignore
    import azure_client  # type: ignore

# Minimal default catalog to populate dropdown without Azure
DEFAULT_EXERCISES = [
    {
        "name": "Warrior II",
        "metadata_json": {
            "tolerances": {
                "left_knee_target": 90, "left_knee": 15,
                "right_knee_target": 90, "right_knee": 15,
                "torso_tilt_target": 0, "torso_tilt": 15,
            }
        },
        "default_tolerance": 15.0,
    },
    {
        "name": "Tree Pose",
        "metadata_json": {
            "tolerances": {
                "left_knee_target": 45, "left_knee": 20,
                "right_knee_target": 180, "right_knee": 20,
                "torso_tilt_target": 0, "torso_tilt": 15,
            }
        },
        "default_tolerance": 20.0,
    },
    {
        "name": "Chair Pose",
        "metadata_json": {
            "tolerances": {
                "left_knee_target": 100, "left_knee": 15,
                "right_knee_target": 100, "right_knee": 15,
                "torso_tilt_target": 10, "torso_tilt": 15,
            }
        },
        "default_tolerance": 15.0,
    },
    {
        "name": "Squat",
        "metadata_json": {
            "tolerances": {
                "left_knee_target": 100, "left_knee": 15,
                "right_knee_target": 100, "right_knee": 15,
                "torso_tilt_target": 10, "torso_tilt": 15,
            }
        },
        "default_tolerance": 15.0,
    },
    {
        "name": "Lunge",
        "metadata_json": {
            "tolerances": {
                "left_knee_target": 90, "left_knee": 15,
                "right_knee_target": 160, "right_knee": 20,
                "torso_tilt_target": 5, "torso_tilt": 15,
            }
        },
        "default_tolerance": 15.0,
    },
    {
        "name": "Push-up",
        "metadata_json": {
            "tolerances": {
                "left_elbow_target": 160, "left_elbow": 20,
                "right_elbow_target": 160, "right_elbow": 20,
                "torso_tilt_target": 0, "torso_tilt": 10,
            }
        },
        "default_tolerance": 20.0,
    },
]

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_LANGUAGE_CODES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "mr": "mr-IN",
    "kn": "kn-IN",
    "ta": "ta-IN",
    "ml": "ml-IN",
    # Bulbul does not list Bhojpuri as a supported target language. The app
    # uses its Hindi voice for Bhojpuri text as a best-effort fallback.
    "bho": "hi-IN",
}
SARVAM_NATIVE_LANGUAGES = ["en", "hi", "mr", "kn", "ta", "ml"]
SUPPORTED_TTS_LANGUAGES = tuple(SARVAM_LANGUAGE_CODES)


def _tts_configuration() -> dict:
    sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()
    if sarvam_key:
        return {
            "provider": "Sarvam Bulbul v3",
            "configured": True,
            # Sarvam does not provide a TTS health probe; actual access is
            # confirmed when a user previews or requests a spoken cue.
            "reachable": None,
            "languages": SARVAM_NATIVE_LANGUAGES,
            "fallback_languages": ["bho"],
            "mode": "sarvam",
        }

    speech_url = os.getenv("TTS_API_URL", "").strip()
    if not speech_url:
        return {
            "provider": None,
            "configured": False,
            "reachable": False,
            "languages": [],
            "fallback_languages": [],
            "mode": None,
        }

    health_url = os.getenv("TTS_HEALTH_URL", "").strip()
    if not health_url:
        health_url = re.sub(r"/v1/audio/speech/?$", "/health", speech_url)
    reachable = False
    if health_url and health_url != speech_url:
        try:
            health_headers = {}
            health_api_key = os.getenv("TTS_API_KEY", "").strip()
            if health_api_key:
                health_headers["Authorization"] = f"Bearer {health_api_key}"
            reachable = requests.get(health_url, headers=health_headers, timeout=(1.5, 2.5)).ok
        except requests.RequestException:
            reachable = False
    return {
        "provider": "Svara-compatible TTS",
        "configured": True,
        "reachable": reachable,
        "languages": list(SUPPORTED_TTS_LANGUAGES),
        "fallback_languages": [],
        "mode": "openai-compatible",
    }


def create_app() -> Flask:
    app = Flask(__name__)
    allowed_origins = [origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if origin.strip()]
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}}, supports_credentials=False)

    production = os.getenv("APP_ENV", "development").lower() == "production"
    secret_key = os.getenv("FLASK_SECRET_KEY", "").strip()
    if production and len(secret_key) < 32:
        raise RuntimeError("FLASK_SECRET_KEY must be set to at least 32 characters in production.")
    app.config["SECRET_KEY"] = secret_key or "pose-coach-local-development-only"
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", str(8 * 1024 * 1024)))
    token_serializer = URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="pose-coach-auth-v1")
    token_max_age = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", str(30 * 24 * 60 * 60)))

    database_url = os.getenv("DATABASE_URL", "sqlite:///dev.db")
    init_engine_and_session(database_url)

    # Seed default exercises if DB is empty
    if db_session.query(Exercise).count() == 0:
        for item in DEFAULT_EXERCISES:
            ex = Exercise(
                name=item["name"],
                metadata_json=item.get("metadata_json"),
                default_tolerance=item.get("default_tolerance", 15.0),
            )
            db_session.add(ex)
        db_session.commit()

    analyzer = PoseAnalyzer()

    def _require_auth():
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        token = auth.split(" ", 1)[1]
        try:
            payload = token_serializer.loads(token, max_age=token_max_age)
        except (BadSignature, SignatureExpired):
            return None
        return payload.get("user_id") if isinstance(payload, dict) else None

    @app.post("/api/v1/auth/register")
    def register():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = (data.get("password") or "").strip()
        email = (data.get("email") or "").strip() or None
        if not username or not password:
            return jsonify({"error": "username and password required"}), 400
        if db_session.query(User).filter_by(username=username).first():
            return jsonify({"error": "username taken"}), 409
        if email and db_session.query(User).filter_by(email=email).first():
            return jsonify({"error": "email already registered"}), 409
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            email=email,
            name=(data.get("name") or "").strip() or None,
        )
        db_session.add(user)
        db_session.commit()
        return jsonify({"id": user.id}), 201

    @app.post("/api/v1/auth/login")
    def login():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = (data.get("password") or "").strip()
        user = db_session.query(User).filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "invalid credentials"}), 401
        token = token_serializer.dumps({"user_id": user.id})
        return jsonify({"token": token, "user": {"id": user.id, "username": user.username, "email": user.email, "name": user.name, "height_cm": user.height_cm, "weight_kg": user.weight_kg}})

    @app.get("/api/v1/profile")
    def get_profile():
        user_id = _require_auth()
        if not user_id:
            return jsonify({"error": "unauthorized"}), 401
        user = db_session.get(User, int(user_id))
        if not user:
            return jsonify({"error": "not found"}), 404
        return jsonify({"id": user.id, "username": user.username, "email": user.email, "name": user.name, "height_cm": user.height_cm, "weight_kg": user.weight_kg})

    @app.put("/api/v1/profile")
    def update_profile():
        user_id = _require_auth()
        if not user_id:
            return jsonify({"error": "unauthorized"}), 401
        data = request.get_json(silent=True) or {}
        user = db_session.get(User, int(user_id))
        if not user:
            return jsonify({"error": "not found"}), 404
        user.name = data.get("name", user.name)
        user.email = data.get("email", user.email)
        user.height_cm = data.get("height_cm", user.height_cm)
        user.weight_kg = data.get("weight_kg", user.weight_kg)
        db_session.commit()
        return jsonify({"ok": True})

    @app.get("/api/v1/health")
    def health() -> tuple:
        speech = _tts_configuration()
        return jsonify({"status": "ok", "speech_configured": speech["configured"], "speech_provider": speech["provider"]}), 200

    @app.get("/api/v1/tts/status")
    def tts_status() -> tuple:
        configuration = _tts_configuration()
        voices = [f"{language}_{gender}" for language in SUPPORTED_TTS_LANGUAGES for gender in ("female", "male")]
        return jsonify({**configuration, "voices": voices}), 200

    @app.post("/api/v1/tts/speech")
    def synthesize_speech():
        if not _require_auth():
            return jsonify({"error": "Sign in to use spoken coaching."}), 401
        configuration = _tts_configuration()
        if not configuration["configured"]:
            return jsonify({"error": "Natural speech is not configured. Add SARVAM_API_KEY or configure TTS_API_URL on the server."}), 503

        data = request.get_json(silent=True) or {}
        spoken_text = str(data.get("input", "")).strip()
        voice = str(data.get("voice", "")).strip().lower()
        allowed_voices = {
            f"{language}_{gender}"
            for language in SUPPORTED_TTS_LANGUAGES
            for gender in ("female", "male")
        }
        allowed_voices.update({"bho_female", "bho_male"})
        if not spoken_text or len(spoken_text) > 600:
            return jsonify({"error": "Speech text must contain 1 to 600 characters."}), 400
        if voice not in allowed_voices:
            return jsonify({"error": "The selected language voice is not supported."}), 400

        language, gender = voice.rsplit("_", 1)
        if configuration["mode"] == "sarvam":
            api_key = os.getenv("SARVAM_API_KEY", "").strip()
            payload = {
                "model": os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
                "text": spoken_text,
                "language_code": SARVAM_LANGUAGE_CODES[language],
                "speaker": "shubh" if gender == "male" else "priya",
                "pace": 0.94,
                "speech_sample_rate": 22050,
            }
            try:
                upstream = requests.post(
                    os.getenv("SARVAM_TTS_URL", SARVAM_TTS_URL).strip(),
                    json=payload,
                    headers={"Content-Type": "application/json", "api-subscription-key": api_key},
                    timeout=(5, 60),
                )
            except requests.RequestException:
                app.logger.exception("Sarvam text-to-speech request failed")
                return jsonify({"error": "Could not connect to the configured voice provider."}), 502
            if not upstream.ok:
                app.logger.warning("Sarvam text-to-speech returned status %s", upstream.status_code)
                return jsonify({"error": "The voice provider rejected this request. Check the API key and selected language."}), 502
            try:
                audio_items = upstream.json().get("audios") or []
                audio_bytes = base64.b64decode(audio_items[0], validate=True)
                if not audio_bytes:
                    raise ValueError("empty audio")
            except (ValueError, IndexError, TypeError, requests.exceptions.JSONDecodeError):
                app.logger.warning("Sarvam text-to-speech returned an invalid audio response")
                return jsonify({"error": "The voice provider returned an invalid audio response."}), 502
            return Response(
                audio_bytes,
                mimetype="audio/wav",
                headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            )

        upstream_url = os.getenv("TTS_API_URL", "").strip()
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("TTS_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": os.getenv("TTS_MODEL", "svara-tts-v1"),
            "input": spoken_text,
            "voice": voice,
            "response_format": "mp3",
        }
        try:
            upstream = requests.post(upstream_url, json=payload, headers=headers, timeout=(5, 60))
        except requests.RequestException:
            app.logger.exception("Text-to-speech service request failed")
            return jsonify({"error": "Could not connect to the configured speech service."}), 502
        if not upstream.ok:
            app.logger.warning("Text-to-speech service returned status %s", upstream.status_code)
            return jsonify({"error": "The configured speech service rejected this voice request."}), 502
        content_type = upstream.headers.get("Content-Type", "audio/mpeg").split(";", 1)[0]
        if not content_type.startswith("audio/"):
            return jsonify({"error": "The speech service did not return an audio file."}), 502
        return Response(
            upstream.content,
            mimetype=content_type,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/v1/exercises")
    def list_exercises() -> tuple:
        exercise_id = request.args.get("exercise_id")
        if exercise_id:
            exercise = db_session.get(Exercise, int(exercise_id))
            if exercise is None:
                try:
                    template = azure_client.get_exercise_template(exercise_id)
                except Exception:
                    template = None
                if template is None:
                    return jsonify({"error": "Exercise not found"}), 404
                return jsonify({"exercise": template}), 200

            return jsonify({
                "exercise": {
                    "id": exercise.id,
                    "name": exercise.name,
                    "metadata_json": exercise.metadata_json,
                    "default_tolerance": exercise.default_tolerance,
                }
            }), 200

        rows = db_session.query(Exercise).all()
        payload = [
            {
                "id": r.id,
                "name": r.name,
                "metadata_json": r.metadata_json,
                "default_tolerance": r.default_tolerance,
            }
            for r in rows
        ]
        return jsonify({"exercises": payload}), 200

    def _decode_image_to_rgb_ndarray(image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)

    def _read_frame_from_request() -> np.ndarray:
        if request.content_type and request.content_type.startswith("application/json"):
            data = request.get_json(silent=True) or {}
            image_b64 = data.get("image_base64")
            if not image_b64:
                raise ValueError("Missing image_base64 in JSON body")
            if image_b64.startswith("data:"):
                image_b64 = image_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)
            return _decode_image_to_rgb_ndarray(image_bytes)

        if "frame" not in request.files:
            raise ValueError("Missing 'frame' file in form-data")
        file_storage = request.files["frame"]
        _ = secure_filename(file_storage.filename) if file_storage.filename else "frame.jpg"
        image_bytes = file_storage.read()
        return _decode_image_to_rgb_ndarray(image_bytes)

    @app.post("/api/v1/pose/analyze")
    def analyze_pose() -> tuple:
        if not _require_auth():
            return jsonify({"error": "Sign in to use pose analysis."}), 401
        try:
            image_rgb = _read_frame_from_request()
        except Exception as exc:
            return jsonify({"error": f"Invalid image: {exc}"}), 400

        exercise_id = request.args.get("exercise_id")
        reference_template = None
        if exercise_id:
            exercise = None
            # Accept numeric id or name slug (e.g., "pushup")
            try:
                exercise = db_session.get(Exercise, int(exercise_id))
            except Exception:
                exercise = db_session.query(Exercise).filter(Exercise.name.ilike(exercise_id.replace('-', ' '))).first()
            if exercise is not None and exercise.metadata_json:
                reference_template = {
                    **exercise.metadata_json,
                    "exercise_name": exercise.name,
                }
            else:
                try:
                    reference_template = azure_client.get_exercise_template(exercise_id)
                except Exception:
                    reference_template = None

        result = analyzer.analyze(image_rgb, reference_template)
        return jsonify(result), 200

    @app.post("/api/v1/logs")
    def create_log() -> tuple:
        data = request.get_json(silent=True) or {}
        user_id_from_token = _require_auth()
        if not user_id_from_token:
            return jsonify({"error": "unauthorized"}), 401
        try:
            exercise_id = int(data.get("exercise_id")) if data.get("exercise_id") is not None else None
            user_id = user_id_from_token
            score = float(data.get("score", 0))
            suggestions_json = data.get("suggestions", [])
            landmarks_json = data.get("landmarks", [])
        except Exception as exc:
            return jsonify({"error": f"Invalid payload: {exc}"}), 400

        attempt = AttemptLog(
            exercise_id=exercise_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            score=score,
            suggestions_json=suggestions_json,
            landmarks_json=landmarks_json,
        )
        db_session.add(attempt)
        db_session.commit()

        return jsonify({"id": attempt.id}), 201

    @app.get("/api/v1/logs")
    def list_logs() -> tuple:
        user_id = _require_auth()
        if not user_id:
            return jsonify({"error": "unauthorized"}), 401
        q = db_session.query(AttemptLog)
        if user_id:
            q = q.filter(AttemptLog.user_id == int(user_id))
        rows = q.order_by(AttemptLog.timestamp.desc()).limit(200).all()
        payload = [
            {
                "id": r.id,
                "exercise_id": r.exercise_id,
                "exercise": r.exercise.name if r.exercise else None,
                "user_id": r.user_id,
                "timestamp": r.timestamp.isoformat() + "Z",
                "score": r.score,
                "suggestions": r.suggestions_json,
                "landmarks": r.landmarks_json,
            }
            for r in rows
        ]
        return jsonify({"logs": payload}), 200

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        try:
            db_session.remove()
        except Exception:
            pass

    @app.errorhandler(413)
    def payload_too_large(_error):
        return jsonify({"error": "Request is too large."}), 413

    return app


# Vercel's Flask runtime discovers the application through a module-level
# Flask instance named `app`. Render/Gunicorn can use the same WSGI instance.
app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("APP_ENV", "development").lower() != "production")

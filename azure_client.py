import os
from typing import Any, Dict, Optional

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_RESOURCE_NAME = os.getenv("AZURE_RESOURCE_NAME", "")


def _azure_configured() -> bool:
    return bool(AZURE_ENDPOINT and AZURE_API_KEY and AZURE_RESOURCE_NAME)


def get_exercise_template(exercise_id: str) -> Optional[Dict[str, Any]]:
    """Fetch exercise template from Azure.
    This is a stub that returns None unless credentials are provided.
    """
    if not _azure_configured():
        return None
    # TODO: Implement actual Azure call. Keep credentials external.
    # Return a minimal placeholder structure to illustrate format.
    return {
        "id": exercise_id,
        "name": f"Exercise-{exercise_id}",
        "reference_landmarks": [],
        "tolerances": {
            "elbow_angle": 15.0,
            "knee_angle": 15.0,
            "shoulder_angle": 15.0,
        },
        "metadata": {},
    }


def get_supplementary_guidance(exercise_id: str) -> Optional[str]:
    if not _azure_configured():
        return None
    # TODO: Implement Azure guidance fetch
    return None

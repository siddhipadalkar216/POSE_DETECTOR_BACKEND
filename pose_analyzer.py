from typing import Any, Dict, List, Optional, Tuple
import math
import os

import numpy as np
from PIL import Image, ImageDraw

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None  # Allow environment without mediapipe at import time


class PoseAnalyzer:
    """Analyze human pose given an RGB image using MediaPipe Pose or optional MMPose.

    If USE_MMPOSE=1 and MMPOSE_CONFIG/MMPOSE_CHECKPOINT are set, uses MMPose.
    Otherwise defaults to MediaPipe.
    """

    def __init__(self) -> None:
        self._backend = os.getenv("USE_MMPOSE", "0") == "1"
        self._mp_pose = None
        self._pose = None
        self._mmpose_infer = None

        if self._backend:
            self._try_init_mmpose()
        else:
            self._try_init_mediapipe()

    def _try_init_mediapipe(self) -> None:
        if mp is None:
            return
        self._mp_pose = mp.solutions.pose
        try:
            model_complexity = int(os.getenv("MEDIAPIPE_MODEL_COMPLEXITY", "1"))
        except ValueError:
            model_complexity = 1
        model_complexity = max(0, min(2, model_complexity))
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    def _try_init_mmpose(self) -> None:  # pragma: no cover - optional dependency
        try:
            from mmpose.apis import init_model, inference_topdown
            from mmpose.structures import merge_data_samples
            from mmpose.apis import MMPoseInferencer
        except Exception:
            self._backend = False
            self._try_init_mediapipe()
            return
        config = os.getenv("MMPOSE_CONFIG", "")
        checkpoint = os.getenv("MMPOSE_CHECKPOINT", "")
        device = os.getenv("MMPOSE_DEVICE", "cpu")
        if not (config and checkpoint):
            self._backend = False
            self._try_init_mediapipe()
            return
        try:
            self._mmpose_infer = MMPoseInferencer(pose2d=config, pose2d_weights=checkpoint, device=device)
        except Exception:
            self._backend = False
            self._try_init_mediapipe()

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        dot = float(np.dot(ba, bc))
        norm = float(np.linalg.norm(ba) * np.linalg.norm(bc))
        if norm == 0:
            return 0.0
        cosang = max(min(dot / norm, 1.0), -1.0)
        return math.degrees(math.acos(cosang))

    @staticmethod
    def _to_points(landmarks: List[Dict[str, float]], indices: Tuple[int, int, int], aspect_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ia, ib, ic = indices
        a = np.array([landmarks[ia]["x"] * aspect_ratio, landmarks[ia]["y"]], dtype=float)
        b = np.array([landmarks[ib]["x"] * aspect_ratio, landmarks[ib]["y"]], dtype=float)
        c = np.array([landmarks[ic]["x"] * aspect_ratio, landmarks[ic]["y"]], dtype=float)
        return a, b, c

    @staticmethod
    def _compute_key_angles(landmarks: List[Dict[str, float]], aspect_ratio: float = 16 / 9) -> Dict[str, float]:
        RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 12, 14, 16
        LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 11, 13, 15
        RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
        LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
        angles: Dict[str, float] = {}

        def visible(indices: Tuple[int, ...]) -> bool:
            return all(
                index < len(landmarks)
                and float(landmarks[index].get("visibility", 1.0)) >= 0.4
                for index in indices
            )

        for name, triplet in [
            ("right_elbow", (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)),
            ("left_elbow", (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)),
            ("right_knee", (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)),
            ("left_knee", (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)),
        ]:
            try:
                if visible(triplet):
                    a, b, c = PoseAnalyzer._to_points(landmarks, triplet, aspect_ratio)
                    angles[name] = PoseAnalyzer._angle(a, b, c)
            except Exception:
                pass

        try:
            # Zero degrees means the shoulder-to-hip line is vertical.
            if visible((LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)):
                shoulder_mid = np.array([
                    (landmarks[LEFT_SHOULDER]["x"] + landmarks[RIGHT_SHOULDER]["x"]) * aspect_ratio / 2,
                    (landmarks[LEFT_SHOULDER]["y"] + landmarks[RIGHT_SHOULDER]["y"]) / 2,
                ])
                hip_mid = np.array([
                    (landmarks[LEFT_HIP]["x"] + landmarks[RIGHT_HIP]["x"]) * aspect_ratio / 2,
                    (landmarks[LEFT_HIP]["y"] + landmarks[RIGHT_HIP]["y"]) / 2,
                ])
                torso = hip_mid - shoulder_mid
                angles["torso_tilt"] = math.degrees(math.atan2(abs(float(torso[0])), abs(float(torso[1]))))
        except Exception:
            pass

        for name, first, second in [
            ("shoulder_tilt", LEFT_SHOULDER, RIGHT_SHOULDER),
            ("hip_tilt", LEFT_HIP, RIGHT_HIP),
        ]:
            try:
                if visible((first, second)):
                    dx = abs((landmarks[first]["x"] - landmarks[second]["x"]) * aspect_ratio)
                    dy = abs(landmarks[first]["y"] - landmarks[second]["y"])
                    angles[name] = math.degrees(math.atan2(dy, max(dx, 1e-6)))
            except Exception:
                pass

        knees = [angles[key] for key in ("left_knee", "right_knee") if key in angles]
        if knees:
            angles["knee_average"] = sum(knees) / len(knees)
        if len(knees) == 2:
            angles["front_knee"] = min(knees)
            angles["back_knee"] = max(knees)
        elbows = [angles[key] for key in ("left_elbow", "right_elbow") if key in angles]
        if elbows:
            angles["elbow_average"] = sum(elbows) / len(elbows)

        shoulder_flexions = []
        for triplet in [(LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST), (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST)]:
            try:
                if visible(triplet):
                    a, b, c = PoseAnalyzer._to_points(landmarks, triplet, aspect_ratio)
                    shoulder_flexions.append(PoseAnalyzer._angle(a, b, c))
            except Exception:
                pass
        if shoulder_flexions:
            angles["shoulder_flexion_average"] = sum(shoulder_flexions) / len(shoulder_flexions)

        try:
            if visible((LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE)):
                shoulder_mid = np.array([
                    (landmarks[LEFT_SHOULDER]["x"] + landmarks[RIGHT_SHOULDER]["x"]) * aspect_ratio / 2,
                    (landmarks[LEFT_SHOULDER]["y"] + landmarks[RIGHT_SHOULDER]["y"]) / 2,
                ])
                hip_mid = np.array([
                    (landmarks[LEFT_HIP]["x"] + landmarks[RIGHT_HIP]["x"]) * aspect_ratio / 2,
                    (landmarks[LEFT_HIP]["y"] + landmarks[RIGHT_HIP]["y"]) / 2,
                ])
                ankle_mid = np.array([
                    (landmarks[LEFT_ANKLE]["x"] + landmarks[RIGHT_ANKLE]["x"]) * aspect_ratio / 2,
                    (landmarks[LEFT_ANKLE]["y"] + landmarks[RIGHT_ANKLE]["y"]) / 2,
                ])
                angles["plank_line"] = PoseAnalyzer._angle(shoulder_mid, hip_mid, ankle_mid)
        except Exception:
            pass

        return angles

    def _extract_landmarks(self, image_rgb: np.ndarray) -> List[Dict[str, float]]:
        if self._backend and self._mmpose_infer is not None:  # MMPose path
            try:  # pragma: no cover - heavy dependency
                # MMPoseInferencer returns a generator of results
                result_gen = self._mmpose_infer(images=image_rgb, show=False, pred_out_dir=None)
                result = next(result_gen)
                # Extract first instance keypoints in normalized [0..1] space
                instances = result.get('predictions', [{}])[0].get('keypoints', [])
                # Heuristic normalization by image size
                h, w = image_rgb.shape[:2]
                lmks: List[Dict[str, float]] = []
                for kp in (instances[0] if instances else []):
                    x, y, score = kp
                    lmks.append({"x": float(x) / max(w, 1), "y": float(y) / max(h, 1), "z": 0.0, "visibility": float(score)})
                return lmks
            except Exception:
                return []

        # MediaPipe path
        if self._pose is None:
            return []
        h, w = image_rgb.shape[:2]
        max_w = 640
        if w > max_w:
            scale = max_w / float(w)
            new_w = max_w
            new_h = int(h * scale)
            image_rgb = self._resize_image(image_rgb, new_w, new_h)
        results = self._pose.process(image_rgb)
        if not results.pose_landmarks:
            return []
        lmks = []
        for lm in results.pose_landmarks.landmark:
            lmks.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "visibility": float(lm.visibility)})
        return lmks

    @staticmethod
    def _resize_image(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        resized = Image.fromarray(img).resize((new_w, new_h), Image.Resampling.LANCZOS)
        return np.asarray(resized)

    def _annotate_image(self, image_rgb: np.ndarray, landmarks: List[Dict[str, float]]) -> str:
        h, w = image_rgb.shape[:2]
        img = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img)
        # connections similar to CanvasOverlay
        connections = [
            (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 12), (23, 24), (11, 23), (12, 24),
            (23, 25), (25, 27), (24, 26), (26, 28),
            (27, 29), (29, 31), (28, 30), (30, 32),
        ]
        for a, b in connections:
            if a < len(landmarks) and b < len(landmarks):
                pa = landmarks[a]; pb = landmarks[b]
                if pa.get("visibility", 1.0) >= 0.4 and pb.get("visibility", 1.0) >= 0.4:
                    draw.line([(pa["x"] * w, pa["y"] * h), (pb["x"] * w, pb["y"] * h)], fill=(0, 200, 255), width=3)
        for lm in landmarks:
            if lm.get("visibility", 1.0) < 0.4:
                continue
            x = lm["x"] * w; y = lm["y"] * h
            r = 3
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(0, 255, 0))
        # encode to base64 JPEG
        import io, base64
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def analyze(self, image_rgb: np.ndarray, reference_template: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        landmarks = self._extract_landmarks(image_rgb)
        height, width = image_rgb.shape[:2]
        angles = self._compute_key_angles(landmarks, width / max(height, 1)) if landmarks else {}
        key_landmarks = [11, 12, 23, 24, 25, 26, 27, 28]
        visible_values = [
            landmarks[index].get("visibility", 1.0)
            for index in key_landmarks
            if index < len(landmarks)
        ]
        pose_confidence = float(np.mean(visible_values)) if visible_values else 0.0
        exercise_name = (reference_template.get("exercise_name") or "").lower() if isinstance(reference_template, dict) else ""
        if "warrior" in exercise_name or "lunge" in exercise_name or "tree" in exercise_name:
            has_exercise_angles = "left_knee" in angles and "right_knee" in angles
        elif "push" in exercise_name:
            has_exercise_angles = (
                ("left_elbow" in angles or "right_elbow" in angles)
                and "plank_line" in angles
            )
        else:
            has_exercise_angles = any(key in angles for key in ("left_knee", "right_knee", "left_elbow", "right_elbow"))
        pose_detected = (
            len(landmarks) >= 33
            and all(
                index < len(landmarks) and landmarks[index].get("visibility", 1.0) >= 0.35
                for index in (11, 12, 23, 24)
            )
            and has_exercise_angles
        )
        error_joints, suggestions = self._compare_to_reference(angles, reference_template)
        if not pose_detected:
            score = 0.0
            suggestions = ["No clear full-body pose detected. Step back and keep your shoulders, hips and legs visible."]
        elif reference_template:
            tolerances = reference_template.get("tolerances", {}) if isinstance(reference_template, dict) else {}
            if "warrior" in exercise_name or "lunge" in exercise_name:
                target_specs = [("front_knee", 100.0, 25.0), ("back_knee", 160.0, 25.0), ("torso_tilt", 8.0, 22.0)]
            elif "tree" in exercise_name:
                target_specs = [("front_knee", 95.0, 35.0), ("back_knee", 175.0, 25.0), ("shoulder_flexion_average", 165.0, 30.0)]
            elif "push" in exercise_name:
                target_specs = [("elbow_average", 160.0, 25.0), ("plank_line", 180.0, 20.0)]
            else:
                target_specs = [
                    (key, float(tolerances.get(f"{key}_target", 0.0)),
                     float(tolerances.get(f"{key}_angle", tolerances.get(key, 15.0))))
                    for key in angles
                ]
            comparisons = []
            for key, target, tolerance in target_specs:
                measured = angles.get(key)
                if measured is None or target <= 0:
                    continue
                tolerance = max(tolerance, 1.0)
                error = abs(measured - target)
                comparisons.append(1.0 if error <= tolerance else max(0.0, 1.0 - (error - tolerance) / (2 * tolerance)))
            score = float(np.mean(comparisons)) if comparisons else 0.0
        else:
            # Heuristic scoring when no reference provided
            default_sugs = self._default_suggestions(angles)
            penalties = [s for s in default_sugs if s != "Good form"]
            score = max(0.0, 1.0 - 0.25 * len(penalties))
        return {
            "landmarks": landmarks,
            "angles": angles,
            "error_joints": error_joints,
            "suggestions": suggestions if suggestions else self._default_suggestions(angles),
            "overall_score": float(max(0.0, min(1.0, score))),
            "pose_detected": pose_detected,
            "pose_confidence": round(max(0.0, min(1.0, pose_confidence)), 3),
            "backend": "mmpose" if self._backend and self._mmpose_infer is not None else "mediapipe",
            "annotated_image_b64": self._annotate_image(image_rgb, landmarks) if landmarks else None,
        }

    @staticmethod
    def _compare_to_reference(angles: Dict[str, float], reference: Optional[Dict[str, Any]]):
        if not reference:
            return [], []
        tolerances = reference.get("tolerances", {}) if isinstance(reference, dict) else {}
        exercise_name = (reference.get("exercise_name") or "").lower() if isinstance(reference, dict) else ""
        if "warrior" in exercise_name or "lunge" in exercise_name:
            targets = [("front_knee", 100.0, 25.0), ("back_knee", 160.0, 25.0), ("torso_tilt", 8.0, 22.0)]
        elif "tree" in exercise_name:
            targets = [("front_knee", 95.0, 35.0), ("back_knee", 175.0, 25.0), ("shoulder_flexion_average", 165.0, 30.0)]
        elif "push" in exercise_name:
            targets = [("elbow_average", 160.0, 25.0), ("plank_line", 180.0, 20.0)]
        else:
            targets = [
                (key, float(tolerances.get(f"{key}_target", 0.0)),
                 float(tolerances.get(f"{key}_angle", tolerances.get(key, 15.0))))
                for key in angles
            ]
        suggestions: List[str] = []
        error_joints: List[str] = []
        for key, target, tolerance in targets:
            measured = angles.get(key)
            if measured is None or target <= 0:
                continue
            tolerance = max(tolerance, 1.0)
            if abs(measured - target) > tolerance:
                error_joints.append(key)
                direction = "less" if measured > target else "more"
                pretty = key.replace("_", " ")
                suggestions.append(f"Adjust {pretty}: {direction} bend; target {target:.0f}° ±{tolerance:.0f}°")
        return error_joints, suggestions

    @staticmethod
    def _default_suggestions(angles: Dict[str, float]) -> List[str]:
        if not angles:
            return ["No person detected. Ensure you are fully in frame and well lit."]
        tips: List[str] = []
        if angles.get("torso_tilt", 0) > 25:
            tips.append("Keep torso upright")
        if min(angles.get("left_knee", 180), angles.get("right_knee", 180)) < 60:
            tips.append("Avoid excessive knee bend")
        if not tips:
            tips.append("Good form")
        return tips

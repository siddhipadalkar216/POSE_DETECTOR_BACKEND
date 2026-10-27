import numpy as np
from server.pose_analyzer import PoseAnalyzer


def test_angle_right_angle():
  a = np.array([0.0, 0.0])
  b = np.array([0.0, 1.0])
  c = np.array([1.0, 1.0])
  analyzer = PoseAnalyzer()
  angle = analyzer._angle(a, b, c)
  assert 89.0 <= angle <= 91.0


def test_default_suggestions_no_landmarks():
  analyzer = PoseAnalyzer()
  tips = analyzer._default_suggestions({})
  assert isinstance(tips, list) and len(tips) >= 1

from mediapipe.python.solutions.pose import PoseLandmark

from dataclasses import dataclass


@dataclass
class Joint:
    parent_landmark: PoseLandmark
    joint_landmark: PoseLandmark
    child_landmark: PoseLandmark


joint_definitions = [
    (
        PoseLandmark.LEFT_HIP,
        PoseLandmark.LEFT_KNEE,
        PoseLandmark.LEFT_ANKLE,
    ),
    (
        PoseLandmark.RIGHT_HIP,
        PoseLandmark.RIGHT_KNEE,
        PoseLandmark.RIGHT_ANKLE,
    ),
    (
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.RIGHT_ELBOW,
        PoseLandmark.RIGHT_WRIST,
    ),
    (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_ELBOW,
        PoseLandmark.LEFT_WRIST,
    ),
]

JOINTS = {
    target_joint: Joint(parent, target_joint, child)
    for parent, target_joint, child in joint_definitions
}

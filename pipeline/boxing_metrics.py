import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerResult,
)
from mediapipe.python.solutions.pose import PoseLandmark

from pipeline.pipeline import (
    StageBase,
    WorldLandmarkLinearKinematicVariables,
    AngularKinematicVariables,
    JointAngularKinematicVariables,
    BoxingPunchMetrics,
)

from utils.joints import JOINTS, Joint

from typing import List, Tuple


class CalculateBoxingMetrics(
    StageBase[
        WorldLandmarkLinearKinematicVariables, BoxingPunchMetrics
    ]
):
    def execute(
        self, input: WorldLandmarkLinearKinematicVariables
    ) -> BoxingPunchMetrics:
        if input.velocity is not None:
            return BoxingPunchMetrics(
                right_wrist_punching_velocity_magnitude=np.linalg.norm(
                    input.velocity[:, PoseLandmark.RIGHT_WRIST],
                    axis=1,
                ),
                left_wrist_punching_velocity_magnitude=np.linalg.norm(
                    input.velocity[:, PoseLandmark.LEFT_WRIST],
                    axis=1,
                ),
            )
        else:
            raise NotImplementedError

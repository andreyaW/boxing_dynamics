#!/usr/bin/env python3

import click
import logging
from pathlib import Path

from pipeline.pipeline import (
    StageBase,
    VideoConfiguration,
    VideoData,
    LandmarkingStageInput,
    BoxingPunchMetrics,
)
from pipeline.video_loader import VideoLoader
from pipeline.landmarking import ExtractHumanPoseLandmarks
from pipeline.kinematics_extractor import (
    ExtractWorldLandmarkLinearKinematics,
    ExtractJointAngularKinematics,
)

from pipeline.boxing_metrics import CalculateBoxingMetrics

from utils.joints import JOINTS, Joint

import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerOptions,
)
from mediapipe.tasks.python import BaseOptions
from mediapipe.python.solutions.pose import PoseLandmark

from matplotlib import pyplot as plt
import numpy as np


@click.command()
@click.option(
    "--debug-logging",
    is_flag=True,
    help="Enable DEBUG logging",
    default=False,
)
def main(debug_logging: bool):

    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    logging.info("Starting BoxingDynamics pipeline")

    video_config = VideoConfiguration(
        name="Max's Cross Punch",
        path=Path("media/realspeed/cross.MP4"),
    )

    video_data = VideoLoader().execute(video_config)
    landmarkers = ExtractHumanPoseLandmarks().execute(
        LandmarkingStageInput(
            video_data,
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path="assets/pose_landmarker_lite.task"
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                output_segmentation_masks=False,
            ),
        )
    )

    linear_kinematics = (
        ExtractWorldLandmarkLinearKinematics().execute(landmarkers)
    )

    joint_angle_kinematics = ExtractJointAngularKinematics().execute(
        linear_kinematics
    )
    boxing_metrics = CalculateBoxingMetrics().execute(
        linear_kinematics
    )
    fig, axs = plt.subplots(2, sharex=True)
    # fmt: off


    axs[0].plot(boxing_metrics.right_wrist_punching_velocity_magnitude, label='right wrist magnitude', color='r')
    axs[0].plot(boxing_metrics.left_wrist_punching_velocity_magnitude, label='left wrist magnitude', color='b')
    axs[0].set(xlabel="Timestep", ylabel="Velocity", title="Punch Metrics")
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].plot(joint_angle_kinematics.joint_3d_angular_kinematics.position[:, JOINTS[PoseLandmark.RIGHT_KNEE].index], color='r', label='right knee angle')
    axs[1].plot(joint_angle_kinematics.joint_3d_angular_kinematics.position[:, JOINTS[PoseLandmark.LEFT_KNEE].index], color='b', label='left knee angle')
    axs[1].set(xlabel="Timestep", ylabel="Angle", title="Leg load metrics")
    axs[1].grid(True)
    axs[1].legend()
    fig.tight_layout()
    plt.show()

    logging.info("Finished BoxingDynamics pipeline")


if __name__ == "__main__":
    main()

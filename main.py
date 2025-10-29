#!/usr/bin/env python3

import click
import logging
from pathlib import Path

from pipeline.pipeline import (
    StageBase,
    VideoConfiguration,
    VideoData,
    LandmarkingStageInput,
)
from pipeline.video_loader import VideoLoader
from pipeline.landmarking import ExtractHumanPoseLandmarks
from pipeline.kinematics_extractor import (
    ExtractWorldLandmarkLinearKinematics,
    ExtractJointAngularKinematics,
)

import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerOptions,
)
from mediapipe.tasks.python import BaseOptions


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

    pipeline = [
        ExtractWorldLandmarkLinearKinematics(),
        ExtractJointAngularKinematics(),
    ]

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

    linear_kinematics = ExtractWorldLandmarkLinearKinematics().execute(landmarkers)

    joint_angle_kinematics = ExtractJointAngularKinematics().execute(linear_kinematics)


    logging.info("Finished BoxingDynamics pipeline")


if __name__ == "__main__":
    main()

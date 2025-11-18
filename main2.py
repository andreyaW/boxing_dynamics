    #!/usr/bin/env python3

import click
import logging
from pathlib import Path

from pipeline.pipeline import VideoConfiguration, LandmarkingStageInput
from pipeline.video_loader import VideoLoader
from pipeline.landmarking import ExtractHumanPoseLandmarks
from pipeline.add_arrows import AddArrowsToLegs

import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
import cv2


@click.command()
@click.argument(
    "video_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--debug-logging",
    is_flag=True,
    help="Enable DEBUG logging",
    default=False,
)
@click.option(
    "--scale-factor",
    type=float,
    default=None,
    help="Optional scale factor for resizing video frames (e.g. 0.5).",
)
@click.option("--lite", "model_fidelity", flag_value="lite", default="lite", help="Use lite model (default).")
@click.option("--heavy", "model_fidelity", flag_value="heavy", help="Use heavy model.")
def main(video_path: Path, debug_logging: bool, scale_factor: float, model_fidelity):
    """Run only the landmark + arrow overlay stages."""
    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(name)s: %(message)s")

    logging.info(f"Processing video: {video_path}")

    # 1️⃣ Load video
    video_config = VideoConfiguration(name=video_path.stem, path=video_path, scale_factor=scale_factor)
    video_data = VideoLoader().execute(video_config)

    # 2️⃣ Select model (lite/heavy)
    match model_fidelity:
        case "heavy":
            model_asset_path = "assets/pose_landmarker_heavy.task"
        case _:
            model_asset_path = "assets/pose_landmarker_lite.task"

    # 3️⃣ Extract landmarks
    landmarkers = ExtractHumanPoseLandmarks().execute(
        LandmarkingStageInput(
            video_data,
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_asset_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                output_segmentation_masks=False,
            ),
        )
    )

    # 4️⃣ Add arrows
    video_with_arrows = AddArrowsToLegs().execute((video_data, landmarkers))

    logging.info(f"Saving video with arrows to: {output_path}")
    save_video(video_with_arrows, str(output_path))
    logging.info("Done!")

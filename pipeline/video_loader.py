import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

from pathlib import Path
from dataclasses import dataclass

from pipeline.pipeline import StageBase, VideoConfiguration, VideoData


class VideoLoader(StageBase[VideoConfiguration, VideoData]):
    def execute(self, input: VideoConfiguration) -> VideoData:
        cap = cv2.VideoCapture(str(input.path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        msg = "Loading frames from video"
        if input.scale_factor is not None:
            msg += f", downscaling by {input.scale_factor}"
        self.logger.info(msg)

        while cap.isOpened():
            successful, frame = cap.read()

            if not successful:
                self.logger.info(
                    "Can't read frame, breaking out of video read"
                )
                break
            if input.scale_factor is not None:
                frame = cv2.resize(
                    frame,
                    dsize=None,
                    fx=input.scale_factor,
                    fy=input.scale_factor,
                )
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        return VideoData(frames=np.asarray(frames), fps=fps)

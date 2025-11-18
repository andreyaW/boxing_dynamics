"""
add_arrows.py

Draws one arrow on each foot (left and right ankle) for every frame in a video.

Input:
    Tuple[VideoData, List[PoseLandmarkerResult]]
Output:
    VideoData with arrows drawn on both feet
"""

import cv2
import numpy as np
from typing import Tuple, List
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from pipeline.pipeline import StageBase, AddArrowsStageInput, VideoData
from pathlib import Path


class AddArrowsToLegs(
    StageBase[AddArrowsStageInput, VideoData]
):
    """
    Draws simple arrows at the left and right ankle landmarks.
    """

    def __init__(
        self,
        color=(0, 255, 0),
        thickness: int = 3,
        arrow_length: int = 50,
    ):
        """
        Parameters
        ----------
        color : tuple
            BGR color for arrows.
        thickness : int
            Arrow line thickness.
        arrow_length : int
            Length of the foot arrow in pixels.
        """
        self.color = color
        self.thickness = thickness
        self.arrow_length = arrow_length

    def execute(self, input: AddArrowsStageInput ) -> VideoData:

        video_data = input.video_data 
        landmark_results = input.landmarkers
        save_video = input.save_video

        new_frames = []
        height, width, _ = video_data.frames[0].frame.shape

        for frame_idx, frame_data in enumerate(video_data.frames):

            frame = frame_data.frame.copy()
            
            # Get ankle coordinates in pixels
            pts = self._extract_ankles(landmark_results[frame_idx].pose_landmarks, width, height)

            # Draw one arrow per ankle (pointing slightly forward)
            for name, point in pts.items():
                if point is not None:
                    pt_start = tuple(point)
                    pt_end = (int(point[0] + self.arrow_length), int(point[1]))  # forward arrow
                    cv2.arrowedLine(
                        frame,
                        pt_start,
                        pt_end,
                        self.color,
                        self.thickness,
                        tipLength=0.3,
                    )

            new_frame_data = type(frame_data)(
                frame = frame, 
                timestamp_ms=frame_data.timestamp_ms
            )
            new_frames.append(new_frame_data)

        # overwrite old video_data 
        video_data = type(video_data)(
            frames = new_frames, 
            fps = video_data.fps,
            config=video_data.config)

        if save_video: 
            output_path = self.save_video(video_data)
            return video_data, output_path
        else:
            return video_data

    def _extract_ankles(self, pose_landmarks, width: int, height: int):
        """
        Extract left/right ankle coordinates (in pixels) from pose landmarks.
        """
        print(type(pose_landmarks[0]))
        print(pose_landmarks[0][0])
        
        MP_POSE = {"left_ankle": 27, "right_ankle": 28}
        pts = {}

        for name, idx in MP_POSE.items():
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                pts[name] = np.array([int(lm.x * width), int(lm.y * height)])
            else:
                pts[name] = None
        return pts
    
    def save_video(self, video_data):
        """Helper: save VideoData frames to MP4."""
        height, width, _ = video_data.frames[0].frame.shape
        fps = 15  # or video_data.config.fps if available

        video_path = video_data.config.path
        output_path = Path("output") / f"{video_path.stem}_feet_arrows.mp4"
        output_path.parent.mkdir(exist_ok=True)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        for frame_data in video_data.frames:
            writer.write(frame_data.frame)
        writer.release()
        
        return output_path
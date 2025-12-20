import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from mediapipe.python.solutions.pose import PoseLandmark

from matplotlib.animation import FuncAnimation
from pipeline.pipeline import (
    StageBase,
    VideoData,
    BoxingPunchMetrics,
    JointAngularKinematicVariables,
    WorldLandmarkLinearKinematicVariables,
)
from utils.joints import JOINTS, Joint
from utils.videoProcessingFunctions import draw_landmarks_on_image
from typing import Tuple, List
from pathlib import Path
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerResult,
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from mediapipe.python.solutions.pose import PoseLandmark

class FuseVideoAndBoxingMetrics(StageBase):

    def execute(self, input: Tuple[VideoData, BoxingPunchMetrics, List]):
        video_data, boxing_metrics, landmarkers = input
        num_frames = len(video_data.frames)

        # ---------------------------
        # Figure layout: 2 rows x 4 columns
        # Video left, 2x2 gauges center, COM animation right
        # ---------------------------
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(3, 4, width_ratios=[2, 1, 1, 2], hspace=0.4, wspace=0.4)
        ax_video = self.init_video_axes(fig, gs, video_data)
        gauge_axes, needles, center_texts, metrics = self.init_gauge_axes(fig, gs, boxing_metrics)
        ax_com, com_marker, com_path_line, hip_line, shoulder_line, heel_line = self.init_com_axes(fig, gs, boxing_metrics)

        #--------------------------
        # Helper function
        #--------------------------
        theta_start, theta_end = np.pi, 0

        def value_to_angle(val):
            return theta_start + (theta_end - theta_start) * (val)

        def update(frame_idx):
            # Update video frame
            frame_rgb = cv2.cvtColor(video_data.frames[frame_idx].frame, cv2.COLOR_BGR2RGB)
            ax_video.images[0].set_data(frame_rgb)
            ax_video.set_title(f"Frame {frame_idx+1}/{num_frames}")

            # Update gauges
            for i, (needle, text) in enumerate(zip(needles, center_texts)):
                val = metrics[i][frame_idx]  # Correctly index each gauge
                angle = value_to_angle(val)
                needle.set_data([angle, angle], [0, 3.4])
                # text.set_text(f"{int(val*100)}")

            # Update COM marker + path
            com_marker.set_data([boxing_metrics.center_of_mass[frame_idx, 0]],
                                [boxing_metrics.center_of_mass[frame_idx, 2]])
            com_path_line.set_data(boxing_metrics.center_of_mass[:frame_idx+1, 0],
                                boxing_metrics.center_of_mass[:frame_idx+1, 2])

            # Update body markers
            x_h, z_h = self.get_positions(boxing_metrics, frame_idx, 'hip_position')
            hip_line.set_data(x_h, z_h)
            x_s, z_s = self.get_positions(boxing_metrics, frame_idx, 'shoulder_position')
            shoulder_line.set_data(x_s, z_s)
            x_f, z_f = self.get_positions(boxing_metrics, frame_idx, 'heel_position')
            heel_line.set_data(x_f, z_f)

            return [ax_video.images[0], com_marker, com_path_line, hip_line, shoulder_line, heel_line] + needles + center_texts


        anim = FuncAnimation(fig, update, frames=num_frames, interval=int(1e3/video_data.fps), blit=True)
        return anim

    # ---------------------------
    # Functions for each element in the output
    # ---------------------------
    def init_video_axes(self, fig, gs, video_data):
        ax = fig.add_subplot(gs[:, 0])
        frame_rgb = cv2.cvtColor(video_data.frames[0].frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        ax.axis("off")
        return ax

    def init_gauge_axes(self, fig, gs, boxing_metrics):
        """
        Initialize 2x2 gauges for Left/Right Wrist Velocity and Shoulder/Hip Rotation.
        Returns axes, needles, center_texts, and metrics list.
        """
        # Gauge settings
        colors = ["lightcoral", "khaki", "mediumseagreen"]
        section_labels = ["Low", "Medium", "High"]
        # colors = ["mediumseagreen", "darkseagreen", "khaki", "burlywood", "lightcoral"]
        # section_labels = ["Great", "Good", "Okay", "Poor", "Bad"]
        theta_start, theta_end = np.pi, 0
        angles = np.linspace(theta_start, theta_end, len(colors), endpoint=False)
        width = (theta_end - theta_start) / len(colors)

        # 2x2 grid in the center
        gauge_axes = [
            fig.add_subplot(gs[0, 1], projection='polar'),
            fig.add_subplot(gs[1, 1], projection='polar'),
            fig.add_subplot(gs[0, 2], projection='polar'),
            fig.add_subplot(gs[1, 2], projection='polar')
        ]
        gauge_titles = ["Left Wrist Velocity", "Right Wrist Velocity",
                        "Shoulder Rotation", "Hip Rotation"]

        # Metrics for each gauge: must be 1D arrays (num_frames,)
        metrics = [
            boxing_metrics.left_wrist_punching_velocity_magnitude,
            boxing_metrics.right_wrist_punching_velocity_magnitude,
            boxing_metrics.shoulder_rotation_velocity_magnitude,
            boxing_metrics.hip_rotation_velocity_magnitude
        ]

        # Normalize metrics by highest value
        for i in range(len(metrics)):
            metrics[i] = metrics[i]/ np.max(metrics[i])
            
        needles = []
        center_texts = []

        def value_to_angle(val):
            """Convert 0-100 value to angle on gauge."""
            return theta_start + (theta_end - theta_start) * (val)

        # Initialize each gauge
        for ax, title, metric in zip(gauge_axes, gauge_titles, metrics):
            # Draw colored sections
            ax.bar(
                angles,
                height=2,
                width=width,
                bottom=2,
                color=colors,
                edgecolor="white",
                linewidth=2,
                align="edge"
            )
            # Section labels
            for angle, label in zip(angles, section_labels):
                ax.text(angle + width/2, 3.25, label, ha="center", va="center",
                        fontsize=11, fontweight="bold")
            # # Percentage markers
            # for angle, val in zip(list(angles) + [theta_end], np.linspace(0, 100, len(colors)+1)):
            #     ax.text(angle, 4.25, f"{int(val)}%", ha="center", va="center", fontsize=12)

            # Initialize needle at first frame
            initial_val = 0
            initial_angle = value_to_angle(initial_val)
            needle, = ax.plot([initial_angle, initial_angle], [0, 3.4], linewidth=4, color="black")
            # center_text = ax.text(0, 0, f"{int(initial_val)}", ha="center", va="center",
            #                     fontsize=28, fontweight="bold",
            #                     bbox=dict(boxstyle="circle", facecolor="black"), color="white")
            center_text = ax.text(0, 0, "", ha="center", va="center",
                                fontsize=28, fontweight="bold",
                                bbox=dict(boxstyle="circle", facecolor="black"), color="white")

            # Axis styling
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_ylim(0, 4)
            ax.set_axis_off()
            ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

            needles.append(needle)
            center_texts.append(center_text)

        return gauge_axes, needles, center_texts, metrics

    def init_com_axes(self, fig, gs, boxing_metrics):
        ax = fig.add_subplot(gs[:, 3])
        ax.set_title("Center of Mass")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.axis("equal")
        ax.grid(True)

        # COM marker
        com_marker, = ax.plot([], [], color='green', marker='*', ms=15, mec='black', linestyle='None', label='COM')
        # COM trail line
        com_path_line, = ax.plot([], [], color='green', linestyle='-', alpha=0.3)

        # Optional shoulder/hip/heel markers
        x_s, z_s = self.get_positions(boxing_metrics, 0, 'shoulder_position')
        shoulder_line, = ax.plot(x_s, z_s, label="shoulder", color='red')
        x_h, z_h = self.get_positions(boxing_metrics, 0, 'hip_position')
        hip_line, = ax.plot(x_h, z_h, label="hip", color='blue')
        x_f, z_f = self.get_positions(boxing_metrics, 0, 'heel_position')
        heel_line, = ax.plot(x_f, z_f, label="heel", color='black')
        ax.legend()

        return ax, com_marker, com_path_line, hip_line, shoulder_line, heel_line

    def get_positions(self, boxing_metrics, frame_idx, key):
        idx1, idx2 = 0, 2
        pos = getattr(boxing_metrics, key)
        return [pos[frame_idx, idx1, 0], pos[frame_idx, idx1, 1]], [pos[frame_idx, idx2, 0], pos[frame_idx, idx2, 1]]

    def weight_to_distribution_to_visual_bars(self, weight_dist):
        return max(0, 1 - (weight_dist + 1)/2), max(0, (weight_dist + 1)/2)        
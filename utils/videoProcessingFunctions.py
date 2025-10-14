from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import pose as mp_pose

import mediapipe as mp
import numpy as np
import cv2


def get_keypoint_name_from_index(index):
    """Get the name of a keypoint given its index."""
    for name, idx in mp_pose.PoseLandmark.__members__.items():
        if idx.value == index:
            return name
    raise KeyError(f"Index {index} not found in PoseLandmark.")

def get_landmarker_results_from_video(video_path, options, start_time_ms=None, end_time_ms=None) -> list:

    """Process a video file to extract pose landmarks using Mediapipe Pose Landmarker.
    Args:
        video_path: Path to the input video file.
        options: Configuration options for the Mediapipe Pose Landmarker.
        start_time_ms: Optional start time (in milliseconds) to begin processing the video.
        end_time_ms: Optional end time (in milliseconds) to stop processing the video.
    
    Returns:
        A list of dictionaries, each containing:
            - 'timestamp_ms': Timestamp of the frame in milliseconds.
            - 'original_frame': The downscaled video frame (as a NumPy array).
            - 'landmarker_results': The pose landmarker output for that frame.
    """

    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)

    # Get the video frame rate (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{fps=}")

    # If a start time is provided, seek the video to that timestamp (in milliseconds)
    if start_time_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

    # Container for storing pose landmark detection results
    pose_landmarker_results = []

    # Create a Mediapipe pose landmarker using the provided configuration options
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    # Read video frames in a loop until the end
    while cap.isOpened():
        ret, frame = cap.read()  # Grab the next frame
        if not ret:
            # If no frame was read, break out of the loop
            print(f"Can't read frame. Skipping...")
            break

        # Current timestamp of the frame in milliseconds
        curr_frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # If an end time is set and we've reached/passed it, stop processing
        if end_time_ms and curr_frame_timestamp_ms >= end_time_ms:
            break

        # Downscale the frame (reduce resolution by half for efficiency)
        downscaled_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert frame from OpenCV BGR format to Mediapipe SRGB format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2RGB),
        )

        # Run Mediapipe pose detection for this video frame at the given timestamp
        pose_landmarker_result = landmarker.detect_for_video(
            mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        # Store results (timestamp, frame, and landmarker output)
        pose_landmarker_results.append(
            {
                "timestamp_ms": curr_frame_timestamp_ms,
                "original_frame": downscaled_frame,
                "landmarker_results": pose_landmarker_result,
            }
        )

    # Release the video file handle
    cap.release()

    # Return the full list of detection results
    return pose_landmarker_results

def draw_landmarks_on_image(rgb_image, detection_result):
    """ Draws the pose markers on a given rgb image """
    # Extract list of detected pose landmarks from Mediapipe results
    pose_world_landmarks_list = detection_result.pose_world_landmarks

    # Make a copy of the input image so the original isn’t modified
    annotated_image = np.copy(rgb_image)

    # Loop through each set of pose landmarks (there can be more than one person)
    for pose_world_landmarks in pose_world_landmarks_list:
        # Convert the Mediapipe landmarks into a NormalizedLandmarkList proto
        # This is required by the Mediapipe drawing utilities
        pose_world_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_world_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x,  # x-coordinate (normalized [0,1])
                    y=landmark.y,  # y-coordinate (normalized [0,1])
                    z=landmark.z   # z-coordinate (normalized, depth)
                )
                for idx, landmark in enumerate(pose_world_landmarks)
            ]
        )

        # Draw the landmarks and the connections between them
        # - annotated_image: the image to draw on
        # - pose_world_landmarks_proto: landmark points
        # - POSE_CONNECTIONS: predefined skeleton connections
        # - get_default_pose_world_landmarks_style(): default landmark drawing style
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_world_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_world_landmarks_style(),
        )

    # Return the annotated image with pose landmarks drawn
    return annotated_image

def annotate_video(video_path, pose_landmarker_results):
    """ Add media pose tracking to the video """

    first_frame = pose_landmarker_results[0]["original_frame"]
    height, width, _ = first_frame.shape
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    out = cv2.VideoWriter(
        "media/annotated/hook.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),  # width doubled for side-by-side
    )
        
    for frame_data in pose_landmarker_results:
        original_frame = cv2.cvtColor(
            frame_data["original_frame"], cv2.COLOR_BGR2RGB
        )
        landmarks_frame = np.copy(original_frame)
        alpha = 0.5
        landmarks_frame = cv2.addWeighted(
            landmarks_frame,
            alpha,
            np.zeros_like(landmarks_frame),
            1 - alpha,
            0,
        )
        landmarks_frame = draw_landmarks_on_image(
            landmarks_frame, frame_data["landmarker_results"]
        )
        landmarkers = frame_data["landmarker_results"]

        side_by_side = np.concatenate(
            (original_frame, landmarks_frame), axis=1
        )

        side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
        out.write(side_by_side_bgr)
    out.release()


def drawFBD(frame_idx, pose_landmarkers) :
    """ draws a free body diagram on a given frame """

    # add center of mass
    shoulder_mid = (
        (pose_landmarkers[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
         pose_landmarkers[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
        (pose_landmarkers[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
         pose_landmarkers[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
    )

    # Add hip center
    hip_mid = (
        (pose_landmarkers[mp_pose.PoseLandmark.LEFT_HIP.value].x +
         pose_landmarkers[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2,
        (pose_landmarkers[mp_pose.PoseLandmark.LEFT_HIP.value].y +
         pose_landmarkers[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    )

    # draw a point at the center of mass
    cv2.circle(frame_idx, (int(shoulder_mid[0] * frame_idx.shape[1]), int(shoulder_mid[1] * frame_idx.shape[0])), 5, (0, 255, 0), -1)
    cv2.circle(frame_idx, (int(hip_mid[0] * frame_idx.shape[1]), int(hip_mid[1] * frame_idx.shape[0])), 5, (0, 0, 255), -1)

    # # Weight distribution analysis
    # hip_center = [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2]
    # shoulder_center = [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2]
    # balance_offset = abs(hip_center[0] - shoulder_center[0])
    # analysis['weight_distribution'] = 1.0 - min(balance_offset * 10, 1.0)
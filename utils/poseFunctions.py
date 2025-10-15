from mediapipe.python.solutions import pose as mp_pose

import numpy as np
import cv2

def get_3d_pose_human_frame_for_keypoint(pose_world_landmarkers, 
                                         keypoint: mp_pose.PoseLandmark) -> np.ndarray:
    """Get the 3D coordinates of one of the 33 keypoints from pose world landmarks.
    Args:
        pose_world_landmarkers: The pose world landmarks from Mediapipe Pose Landmarker output.
        keypoint: The specific keypoint to extract (e.g., mp_pose.PoseLandmark.LEFT_KNEE).
    Returns:
        A NumPy array with the (x, y, z) coordinates of the specified keypoint.
    """
    return np.array(
        [
            pose_world_landmarkers[keypoint].x,
            pose_world_landmarkers[keypoint].y,
            pose_world_landmarkers[keypoint].z,
        ]
    )

def draw_3d_pose_human_frame(pose_landmarker_result, ax):
    """
    Draw a 3D human pose skeleton with a body-centered reference frame.

    Args:
        pose_landmarker_result (dict): A dictionary containing Mediapipe's 
            `pose_world_landmarks` in 3D world coordinates for a single frame. 
            Expected structure:
                {
                    "landmarker_results": <mediapipe result object>
                }
        ax (matplotlib.axes._subplots.Axes3DSubplot): A 3D matplotlib axis 
            object to plot the skeleton and reference frame on.

    Returns:
        None. (The function draws directly on the provided matplotlib axis.)
    """

    # Extract 3D pose landmarks for the first detected person in this frame
    pose_world_landmarkers = pose_landmarker_result["landmarker_results"].pose_world_landmarks[0]

    # Collect x, y, z coordinates of all landmarks except knees
    # (knees are highlighted separately later)
    x = np.array(
        [
            lm.x
            for lm in pose_world_landmarkers
            if lm != mp_pose.PoseLandmark.LEFT_KNEE
            and lm != mp_pose.PoseLandmark.RIGHT_KNEE
        ]
    )
    y = np.array(
        [
            lm.y
            for lm in pose_world_landmarkers
            if lm != mp_pose.PoseLandmark.LEFT_KNEE
            and lm != mp_pose.PoseLandmark.RIGHT_KNEE
        ]
    )
    z = np.array(
        [
            lm.z
            for lm in pose_world_landmarkers
            if lm != mp_pose.PoseLandmark.LEFT_KNEE
            and lm != mp_pose.PoseLandmark.RIGHT_KNEE
        ]
    )

    # Plot all non-knee landmarks as magenta points
    ax.scatter(x, z, y, s=20, c="magenta")

    # Get and plot the left and right knees separately with a bigger star marker
    left_knee = get_3d_pose_human_frame_for_keypoint(
        pose_world_landmarkers, mp_pose.PoseLandmark.LEFT_KNEE
    )
    right_knee = get_3d_pose_human_frame_for_keypoint(
        pose_world_landmarkers, mp_pose.PoseLandmark.RIGHT_KNEE
    )
    ax.scatter(left_knee[0], left_knee[2], left_knee[1],
               marker="*", s=60, color="blue")
    ax.scatter(right_knee[0], right_knee[2], right_knee[1],
               marker="*", s=60, color="blue")

    # Draw skeleton connections (bones) using Mediapipe's predefined connections
    for connection in mp_pose.POSE_CONNECTIONS:
        start, end = connection
        ax.plot([x[start], x[end]],
                [z[start], z[end]],
                [y[start], y[end]],
                color="indigo")

    # Compute the mid-hip point (used as human body reference frame origin)
    left_hip = get_3d_pose_human_frame_for_keypoint(
        pose_world_landmarkers, mp_pose.PoseLandmark.LEFT_HIP
    )
    right_hip = get_3d_pose_human_frame_for_keypoint(
        pose_world_landmarkers, mp_pose.PoseLandmark.RIGHT_HIP
    )
    hip_mid = (left_hip + right_hip) / 2.0

    # Mark the human reference frame origin (mid-hip) with a green star
    ax.scatter(hip_mid[0], hip_mid[2], hip_mid[1],
               c="green", marker="*", s=100,
               label="human frame origin")

    # Draw coordinate axes at the hip midpoint
    # Red = x-axis, Green = y-axis, Blue = z-axis
    ax.quiver(*hip_mid, 1, 0, 0, color="red", alpha=0.5, linewidth=3, arrow_length_ratio=0.4)
    ax.quiver(*hip_mid, 0, 1, 0, color="green", alpha=0.5, linewidth=3, arrow_length_ratio=0.4)
    ax.quiver(*hip_mid, 0, 0, 1, color="blue", alpha=0.5, linewidth=3, arrow_length_ratio=0.4)

    # Configure axis labels and appearance
    ax.set(xlabel="x - human", ylabel="z - human", zlabel="y - human")
    ax.legend(loc="lower left")
    lim = 0.75
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.invert_zaxis()  # Flip z-axis so orientation matches human coordinate convention

# --------------------- Calculate Functions ---------------------------------
def calculate_limb_vector(pose_world_landmarkers, proximal_kp, distal_kp):
    """Compute the vector from one joint (proximal) to another (distal) in the human reference frame."""
    return get_3d_pose_human_frame_for_keypoint(pose_world_landmarkers, proximal_kp) - get_3d_pose_human_frame_for_keypoint(pose_world_landmarkers, distal_kp)


def calculate_nominal_joint_angle(proximal_vec, distal_vec):
    """Calculate the nominal joint angle between two limb vectors using the arccosine of their cosine similarity."""
    similarity = np.dot(proximal_vec, distal_vec) / (
        np.linalg.norm(proximal_vec) * np.linalg.norm(distal_vec)
    )
    return np.arccos(similarity)


def calculate_left_right_flexion(pose_landmarker_result, joint_name: str):
    """Calculate left and right flexion angles for a specified joint (e.g., 'knee' or 'elbow').
    Args:
        pose_landmarker_result: The output from the Mediapipe Pose Landmarker for a single frame.
        joint_name: The joint to calculate flexion for ('knee' or 'elbow').
    Returns:
        A NumPy array with two elements: [left_flexion_angle, right_flexion_angle] in degrees.
    """
    if joint_name == "knee":
        return calculate_left_right_knee_flexion(pose_landmarker_result)
    elif joint_name == "elbow":
        return calculate_left_right_elbow_flexion(pose_landmarker_result)
    else:
        raise ValueError(f"Unsupported joint name: {joint_name}")


def calculate_left_right_knee_flexion(pose_landmarker_result)-> np.ndarray:
    """
    Calculate left and right knee flexion angles from pose landmarks.
    """
    # Extract 3D pose landmarks (world coordinates) from the pose landmarker output
    pose_world_landmarkers = pose_landmarker_result[
        "landmarker_results"
    ].pose_world_landmarks[0]

    # Helper function: compute the vector from one joint (proximal) to another (distal)
    # in the human reference frame.
    calculate_limb_vector = lambda proximal_kp, distal_kp: get_3d_pose_human_frame_for_keypoint(pose_world_landmarkers, proximal_kp) - get_3d_pose_human_frame_for_keypoint(pose_world_landmarkers, distal_kp)

    # ----------------------------
    # Define limb vectors for shank (knee → ankle)
    left_shank = calculate_limb_vector(
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
    )
    right_shank = calculate_limb_vector(
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    )

    # Define limb vectors for thigh (knee → hip)
    left_thigh = calculate_limb_vector(
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_HIP,
    )
    right_thigh = calculate_limb_vector(
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_HIP,
    )

    # ----------------------------
    # Calculate flexion angle at each knee
    # The angle is computed between the thigh vector and the shank vector.
    left_knee_flex_angle = calculate_nominal_joint_angle(
        left_thigh, left_shank
    )
    right_knee_flex_angle = calculate_nominal_joint_angle(
        right_thigh, right_shank
    )

    # Convert the two knee flexion angles from radians → degrees
    return np.rad2deg(
        np.array([left_knee_flex_angle, right_knee_flex_angle])
    )


def calculate_left_right_elbow_flexion(pose_landmarker_result) -> np.ndarray:
    pose_world_landmarkers = pose_landmarker_result["landmarker_results"].pose_world_landmarks[0]

    # --- Compute mid-hip as human frame origin ---
    left_hip = get_3d_pose_human_frame_for_keypoint(pose_world_landmarkers, mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = get_3d_pose_human_frame_for_keypoint(pose_world_landmarkers, mp_pose.PoseLandmark.RIGHT_HIP)
    hip_mid = (left_hip + right_hip) / 2.0

    # --- Translate all landmark coordinates to human-centered frame ---
    def kp(k):
        return get_3d_pose_human_frame_for_keypoint(pose_world_landmarkers, k) - hip_mid

    # --- Define limb vectors in this local frame ---
    left_forearm = kp(mp_pose.PoseLandmark.LEFT_ELBOW) - kp(mp_pose.PoseLandmark.LEFT_WRIST)
    right_forearm = kp(mp_pose.PoseLandmark.RIGHT_ELBOW) - kp(mp_pose.PoseLandmark.RIGHT_WRIST)

    left_upper_arm = kp(mp_pose.PoseLandmark.LEFT_ELBOW) - kp(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_upper_arm = kp(mp_pose.PoseLandmark.RIGHT_ELBOW) - kp(mp_pose.PoseLandmark.RIGHT_SHOULDER)

    # --- Compute flexion angles ---
    left_angle = calculate_nominal_joint_angle(left_upper_arm, left_forearm)
    right_angle = calculate_nominal_joint_angle(right_upper_arm, right_forearm)

    return np.rad2deg(np.array([left_angle, right_angle]))


# --------------------- Free Body Diagram ---------------------------------

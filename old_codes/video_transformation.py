import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *

MAX_FRAME = 300
FC = "FC2"
fx = 1.71847e+03 if FC == "FC1" else 1.85417e+03
fy = 1.53787e+03 if FC == "FC1" else 1.65964e+03
cx = 6.23459e+02 if FC == "FC1" else 6.22157e+02
cy = 3.76958e+02 if FC == "FC1" else 3.84014e+02
dist_coeffs = np.array([-3.7094196768352616e-01, 1.8626642173380867e-01, 0., 0., 0.]) if FC == "FC1" else \
              np.array([-3.6485240781354333e-01, 1.9904269142924666e-01, 0., 0., 0.])

shoulder1_index = 11
shoulder2_index = 12


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,  # For video
    model_complexity=2,       # Highest accuracy (3D requires complexity=2)
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Input/Output paths
input_video_path = f"../input_videos/{FC}_rtmw.mp4"
output_video_path = f"./outputs/{FC}_MediaPipe_with_3D_camera_frame.mp4"

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer (MP4V codec for .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (2 * width, height))

t_arr = np.empty((1, 3))

landmark_3d_cam_arr = np.empty((MAX_FRAME, 33, 3))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw 2D landmarks on frame
        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0)),  # Joint color
            mp_drawing.DrawingSpec(color=(255, 0, 0))   # Connection color
        )

        # Extract 3D landmarks
        landmarks_3d = np.array(
            [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_world_landmarks.landmark]
        )

        # Extract 2D landmarks
        landmarks_2d = np.array(
            [[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark]
        )

        # Scale 2D landmarks
        landmarks_2d[:,0] *= width
        landmarks_2d[:,1] *= height

        
        # Solve for projection matrix and decompose
        # P = solve_projection_matrix(landmarks_3d, landmarks_2d)
        # R_est, t_est = decompose_projection_matrix_with_fixed_intrinsics(P, fx, fy, cx, cy)

        # landmarks_3d_cam = transform_to_camera_frame(landmarks_3d, R_est, t_est)

        # true_dist = np.linalg.norm(landmarks_3d[shoulder1_index] - landmarks_3d[shoulder2_index])
        # est_dist = np.linalg.norm(landmarks_3d_cam[shoulder1_index] - landmarks_3d_cam[shoulder2_index])
        # print(f"true dist: {true_dist}, est dist: {est_dist}")
        # alpha = true_dist / est_dist
        # print("alpha:", alpha)
        # t_est = alpha * t_est
        R_est, t_est = local_to_camera_transformation(landmarks_3d, landmarks_2d, fx, fy, cx, cy, dist_coeffs)

        landmarks_3d_cam = transform_to_camera_frame(landmarks_3d, R_est, t_est)
        

        if landmarks_3d_cam[9][2] < 0:
            print("NEGATIVE")
            print(landmarks_3d_cam[9][2])
            landmark_3d_cam_arr[frame_count] = landmark_3d_cam_arr[frame_count - 1]
        else:
            t_arr = np.append(t_arr, t_est.reshape(1, 3), axis = 0)
            landmark_3d_cam_arr[frame_count] = landmarks_3d_cam

        # Plot 3D landmarks
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([1, 3])
        ax.set_zlim([-1, 1])

        # Plot connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),  # Shoulders
            (11, 12),  # Hips
            (11, 13), (13, 15), (15, 17),  # Right arm
            (12, 14), (14, 16), (16, 18),  # Left arm
            (11, 23), (12, 24),  # Torso
        ]

        for connection in connections:
            start_idx, end_idx = connection
            ax.plot(
                [landmarks_3d_cam[start_idx, 0], landmarks_3d_cam[end_idx, 0]],
                [landmarks_3d_cam[start_idx, 2], landmarks_3d_cam[end_idx, 2]],
                [-landmarks_3d_cam[start_idx, 1], -landmarks_3d_cam[end_idx, 1]],  # Note: Using z for y-axis
                c = 'b'
            )

        # Plot joints
        ax.scatter(
            landmarks_3d_cam[:, 0],
            landmarks_3d_cam[:, 2],
            -landmarks_3d_cam[:, 1],
            c='b',
            s=20  # Size of the joints
        )
    
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        # ax.view_init(-90, -90, 0)  # Adjust camera angle for better visualization

        # Save plot to image
        plot_filename = f"plot_{frame_count:04d}.png"
        plt.savefig(plot_filename)
        plt.close(fig)

        # Read plot image
        plot_image = cv2.imread(plot_filename)
        plot_image = cv2.resize(plot_image, (width, height))

        # Concatenate original frame and plot image
        combined_frame = np.concatenate((annotated_frame, plot_image), axis=1)

        # Write combined frame to output video
        out.write(combined_frame)

        # Remove the plot image file
        os.remove(plot_filename)
    else:
        # If no landmarks detected, duplicate the frame
        combined_frame = np.concatenate((frame, frame), axis=1)
        out.write(combined_frame)

    frame_count += 1

    if frame_count == MAX_FRAME:
        break

print("mean t:", np.mean(t_arr, axis=0))
print("std t:", np.std(t_arr, axis=0))


# Save landmark_3d_cam_arr to a .npy file
np.save(f'./arrays/landmark_3d_cam_{FC}.npy', landmark_3d_cam_arr)

# Release resources
cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames. Results saved to {output_video_path}")

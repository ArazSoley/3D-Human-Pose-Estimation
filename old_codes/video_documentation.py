import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
input_video_path = "../input_videos/FC2_rtmw.mp4"
output_video_path = "FC2_MediaPipe_with_3D.mp4"

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer (MP4V codec for .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (2 * width, height))

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

        # Plot 3D landmarks
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-0.7, 0.7])
        ax.set_ylim([-0.7, 0.7])
        ax.set_zlim([-0.7, 0.7])

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
                [landmarks_3d[start_idx, 0], landmarks_3d[end_idx, 0]],
                [landmarks_3d[start_idx, 1], landmarks_3d[end_idx, 1]],
                [landmarks_3d[start_idx, 2], landmarks_3d[end_idx, 2]],  # Note: Using z for y-axis
                c = 'b'
            )

        # Plot joints
        ax.scatter(
            landmarks_3d[:, 0],
            landmarks_3d[:, 1],
            landmarks_3d[:, 2],
            c='b',
            s=20  # Size of the joints
        )
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(-90, -90, 0)  # Adjust camera angle for better visualization

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

# Release resources
cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames. Results saved to {output_video_path}")

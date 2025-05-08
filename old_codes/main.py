import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=True,  # For images (not video)
    model_complexity=2,       # 0=Light, 1=Medium, 2=Heavy (use 2 for 3D)
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# Input/Output folders
input_dir = "../input_images/"
output_dir = "output_MediaPipe/"
os.makedirs(output_dir, exist_ok=True)

# Process each image
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Read image
    img_path = os.path.join(input_dir, img_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run MediaPipe Pose
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        print(f"No pose detected in {img_name}")
        continue
    
    # Extract 3D landmarks (in normalized coordinates, relative to hip)
    landmarks_3d = results.pose_world_landmarks
    
    # ----------------------------------
    # (Optional) Visualize 2D pose on image
    # ----------------------------------
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
    
    # Save annotated image
    cv2.imwrite(os.path.join(output_dir, f"2d_{img_name}"), annotated_image)
    
    # ----------------------------------
    # Save 3D landmarks to .npz file
    # ----------------------------------
    landmarks_3d_array = np.array([
        [lmk.x, lmk.y, lmk.z] for lmk in landmarks_3d.landmark
    ])
    
    np.savez(
        os.path.join(output_dir, f"3d_landmarks_{img_name.split('.')[0]}.npz"),
        landmarks=landmarks_3d_array
    )
    
    # ----------------------------------
    # (Optional) Plot 3D landmarks
    # ----------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = landmarks_3d_array[:, 0]
    y = landmarks_3d_array[:, 1]
    z = landmarks_3d_array[:, 2]
    
    # Plot joints
    ax.scatter(x, y, z, c='r', marker='o')
    
    # Plot connections (adjust indices as needed)
    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot(
            [x[start_idx], x[end_idx]],
            [y[start_idx], y[end_idx]],
            [z[start_idx], z[end_idx]],
            c='b'
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(-90, -90)  # Adjust camera angle for better visualization
    plt.savefig(os.path.join(output_dir, f"3d_plot_{img_name}"))
    plt.close()

pose.close()
print("Processing complete!")
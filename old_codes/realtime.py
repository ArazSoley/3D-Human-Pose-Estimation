import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # Reduce complexity for real-time performance
    min_detection_confidence=0.7,
    min_tracking_confidence=0.9
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up matplotlib 3D plot
plt.ion()
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=-70)  # Initial camera angle

# Configure plot limits (adjusted for MediaPipe's coordinate system)
ax.set_xlim3d([-1, 1])
ax.set_ylim3d([-1, 1])
ax.set_zlim3d([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Initialize empty plot elements
scat = ax.scatter([], [], [], c='r', marker='o')
lines = [ax.plot([], [], [], 'b')[0] for _ in mp_pose.POSE_CONNECTIONS]

def update_plot(landmarks):
    """Update 3D plot with new landmarks"""
    if landmarks is None:
        return
    
    # Extract coordinates
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    z = landmarks[:, 2]
    
    # Update scatter plot
    scat._offsets3d = (x, y, z)
    
    # Update connections
    for line, connection in zip(lines, mp_pose.POSE_CONNECTIONS):
        start_idx, end_idx = connection
        line.set_data_3d(
            [x[start_idx], x[end_idx]],
            [y[start_idx], y[end_idx]],
            [z[start_idx], z[end_idx]]
        )
    
    # Dynamically adjust view (optional)
    ax.set_xlim3d([x.min()-0.1, x.max()+0.1])
    ax.set_ylim3d([y.min()-0.1, y.max()+0.1])
    ax.set_zlim3d([z.min()-0.1, z.max()+0.1])
    
    fig.canvas.draw()
    fig.canvas.flush_events()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with MediaPipe
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_world_landmarks:
            # Convert landmarks to numpy array
            landmarks = np.array([[lmk.x, lmk.y, lmk.z] 
                                for lmk in results.pose_world_landmarks.landmark])
            
            # Update 3D plot
            update_plot(landmarks)
            
        # Show webcam feed
        cv2.imshow('Webcam', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    pose.close()
    cv2.destroyAllWindows()
    plt.close()
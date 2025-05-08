"""
Transform 3D points from FC1 to FC2 and plot in 3D
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import savgol_filter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def transform_to_global_frame(landmarks, rotation_matrix, translation_vec):
    # Applying rotation matrix to landmarks
    landmarks = landmarks @ rotation_matrix.T

    # Applying translation
    landmarks += np.array(translation_vec)

    return landmarks

def create_global_video(landmarks1, landmarks2, connections, output_video_path, fps = 25, size = (1280, 720)):

    FRAME_COUNT = landmarks1.shape[0]

    # Video writer (MP4V codec for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    # Create a figure once with size matching the video dimensions
    dpi = 75
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvas(fig)  

    for frame in range(FRAME_COUNT):

        # Clear the axis to redraw the new frame
        ax.cla()

        ax.set_xlim([-1, 1])
        ax.set_ylim([0.5, 2.5])
        ax.set_zlim([-1, 0.5])

        # Plotting the table
        # Draw a circular table as a filled disk
        circle_center = [0, 1.4986, -0.386]  # (x, z, -y) position
        radius = 0.6
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = circle_center[0] + radius * np.cos(theta)
        y_circle = circle_center[1] + radius * np.sin(theta)
        z_circle = np.full_like(x_circle, circle_center[2])

        # Create a filled polygon (disk)
        verts = [list(zip(x_circle, y_circle, z_circle))]
        table = Poly3DCollection(verts, color='sienna', alpha=0.5)
        ax.add_collection3d(table)

        # Plot connections
        for connection in connections:
            start_idx, end_idx = connection
            ax.plot(
                [landmarks1[frame, start_idx, 0], landmarks1[frame, end_idx, 0]],
                [landmarks1[frame, start_idx, 2], landmarks1[frame, end_idx, 2]],
                [-landmarks1[frame, start_idx, 1], -landmarks1[frame, end_idx, 1]],  # Note: Using z for y-axis
                c = 'b'
            )

            ax.plot(
                [landmarks2[frame, start_idx, 0], landmarks2[frame, end_idx, 0]],
                [landmarks2[frame, start_idx, 2], landmarks2[frame, end_idx, 2]],
                [-landmarks2[frame, start_idx, 1], -landmarks2[frame, end_idx, 1]],  # Note: Using z for y-axis
                c = 'g'
            )

        # Plot joints
        ax.scatter(
            landmarks1[frame, :25, 0],
            landmarks1[frame, :25, 2],
            -landmarks1[frame, :25, 1],
            c='b',
            s=20  # Size of the joints
        )
        ax.scatter(
            landmarks2[frame, :25, 0],
            landmarks2[frame, :25, 2],
            -landmarks2[frame, :25, 1],
            c='g',
            s=20  # Size of the joints
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Render the canvas to the in-memory buffer
        canvas.draw()
        # Retrieve the RGBA buffer as a NumPy array
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        # Convert RGBA to RGB (video codecs typically expect 3 channels)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        out.write(img)

    # Cleanup
    plt.close(fig)
    out.release()
    cv2.destroyAllWindows()

def apply_filter(landmarks, window_length, polyorder):
    """
    Apply Savitzky-Golay filter (window_length must be odd, and polyorder < window_length)
    """

    assert window_length % 2 == 1, "window_length must be odd"
    assert polyorder < window_length, "polyorder must be less than window_length"

    smoothed_translations = savgol_filter(landmarks, window_length, polyorder, axis=0)

    return smoothed_translations

if __name__ == '__main__':

    #################### Using the average rotation matrix ####################
    rotation_matrix = np.array([[0.07368957,  0.15444128, -0.98525009],
                                [-0.13508931,  0.98037621,  0.14357358],
                                [0.98808943,  0.12251688,  0.09310687]])

    #################### Using the average translation vector ####################
    translation_vec = np.array([1320.62178736, -181.79693471, 1142.09545961])
    
    # Converting to meters
    translation_vec *= 1e-3

    landmarks_path = './UDIVAv0.5/MediaPipe/lego_train'

    fps = 25
    APPLY_FILTER = True

    connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),  # Shoulders
            (11, 12),  # Hips
            (11, 13), (13, 15), (15, 17),  # Right arm
            (12, 14), (14, 16), (16, 18),  # Left arm
            (11, 23), (12, 24),  # Torso
            (23, 24),
            (15, 17), (15, 19), (15, 21), (17, 19), # Right wrist
            (16, 18), (16, 20), (16, 22), (18, 20) # Left wrist
        ]

    # Iterate through all entries in the dataset directory
    for entry in os.listdir(landmarks_path):
        # Construct the output directory
        output_video_dir = os.path.join("./videos/lego_train/", entry)
        
        if not os.path.exists(output_video_dir):
            
            # Construct the full path to the entry
            entry_path = os.path.join(landmarks_path, entry)
            # Check if the entry is a directory
            if os.path.isdir(entry_path):

                landmarks_FC1_path = os.path.join(entry_path, "FC1_landmark_3d_cam.npy")
                landmarks_FC2_path = os.path.join(entry_path, "FC2_landmark_3d_cam.npy")

                landmarks_FC1 = np.load(landmarks_FC1_path)
                landmarks_FC2 = np.load(landmarks_FC2_path)

                # Transforming FC1 landmarks to FC2 frame
                landmarks_FC1 = transform_to_global_frame(landmarks_FC1, rotation_matrix, translation_vec)

                output_video_path = output_video_dir + "/" + entry + "_landmarker_oneeuro_savgol_11_2" + ".mp4"

                if APPLY_FILTER:
                    landmarks_FC1 = apply_filter(landmarks_FC1, 11, 2)
                    landmarks_FC2 = apply_filter(landmarks_FC2, 11, 2)

                # Create directory if it doesn't exist
                os.makedirs(output_video_dir, exist_ok=True)
                
                print("Creating video for: ", entry)

                create_global_video(landmarks_FC1, landmarks_FC2, connections, output_video_path, fps)

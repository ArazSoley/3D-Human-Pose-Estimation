"""
Transform 3D points from FC1 to FC2 and plot in 3D
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import savgol_filter

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

    for frame in range(FRAME_COUNT):

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1.5])
        ax.set_ylim([0, 2.5])
        ax.set_zlim([-1, 1])

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
                c = 'b',
                zorder = 20
            )

            ax.plot(
                [landmarks2[frame, start_idx, 0], landmarks2[frame, end_idx, 0]],
                [landmarks2[frame, start_idx, 2], landmarks2[frame, end_idx, 2]],
                [-landmarks2[frame, start_idx, 1], -landmarks2[frame, end_idx, 1]],  # Note: Using z for y-axis
                c = 'r',
                zorder = 20
            )

        # Plot joints
        ax.scatter(
            landmarks1[frame, :25, 0],
            landmarks1[frame, :25, 2],
            -landmarks1[frame, :25, 1],
            c='b',
            s=20,  # Size of the joints
            zorder = 20
        )
        ax.scatter(
            landmarks2[frame, :25, 0],
            landmarks2[frame, :25, 2],
            -landmarks2[frame, :25, 1],
            c='b',
            s=20,  # Size of the joints
            zorder = 20
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        # Save plot to image
        plot_filename = f"plot_{frame:04d}.png"
        plt.savefig(plot_filename)
        plt.close(fig)

        # Read plot image
        plot_image = cv2.imread(plot_filename)
        plot_image = cv2.resize(plot_image, size)

         # Write combined frame to output video
        out.write(plot_image)

        # Remove the plot image file
        os.remove(plot_filename)

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

    # # The rotation axis and vector as calculated from calibration
    # rotation_axis = np.array([0.01027114, 0.98935196, 0.14478112])
    # rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    # rotation_angle = -85.873476 * np.pi / 180

    # # Scaling the vector by the rotation angle
    # rotation_axis *= rotation_angle

    # # Converting the vector to a rotation matrix
    # rotation_matrix, _ = cv2.Rodrigues(rotation_axis)

    #################### Using the rotation matrix of the session ####################
    rotation_matrix = np.array([[1.0248006017271942e-01, 1.5928201719510268e-01,
           -9.8189972821325022e-01], [-1.1369102978260752e-01,
           9.8250390051985037e-01, 1.4751418647116760e-01],
           [9.8821667007492131e-01, 9.6515928537931867e-02,
           1.1879599540596802e-01]])

    # # The translation vector as calculated from calibration
    # translation_vec = np.array([1.32062178736, -1.8179693471e-1, 1.14209545961])

    #################### Using the translation vector of the session ####################
    translation_vec = np.array([ 1.3751606712503894e+03, -1.5861831506211050e+02,
           1.0806767616717004e+03 ])
    
    # Converting to meters
    translation_vec *= 1e-3

    # Loading 3D landmarks of Fc1 in camera frame
    landmarks_FC1 = np.load('./arrays/landmark_3d_cam_FC1.npy')

    # Loading 3D landmarks of Fc2 in camera frame
    landmarks_FC2 = np.load('./arrays/landmark_3d_cam_FC2.npy')

    # Transforming FC1 landmarks to FC2 frame
    landmarks_FC1 = transform_to_global_frame(landmarks_FC1, rotation_matrix, translation_vec)

    output_video_path = "./outputs/GlobalMediaPipe_FC2.mp4"
    fps = 25

    connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),  # Shoulders
            (11, 12),  # Hips
            (11, 13), (13, 15), (15, 17),  # Right arm
            (12, 14), (14, 16), (16, 18),  # Left arm
            (11, 23), (12, 24),  # Torso
            (23, 24)
        ]
    
    landmarks_FC1 = apply_filter(landmarks_FC1, 5, 2)
    landmarks_FC2 = apply_filter(landmarks_FC2, 5, 2)
    
    create_global_video(landmarks_FC1, landmarks_FC2, connections, output_video_path, fps)
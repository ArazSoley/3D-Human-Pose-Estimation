import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def transform_to_global_frame(landmarks, rotation_matrix, translation_vec):
    # Applying rotation matrix to landmarks
    landmarks = landmarks @ rotation_matrix.T

    # Applying translation
    landmarks += np.array(translation_vec)

    return landmarks


def convert_to_video(hand_dist, output_video_path, fps = 25, size = (1280, 720)):

    FRAME_COUNT = hand_dist.shape[0]

    # Video writer (MP4V codec for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    for frame in range(FRAME_COUNT):

        x = np.arange(hand_dist.shape[0])

        # Create the plot
        plt.figure(figsize=(6, 4))
        plt.plot(x, hand_dist, linestyle='-', color='b', label="Data")

        plt.scatter(x[frame], hand_dist[frame], color='red', s=100, label="Point")

        # Add text to the upper right corner
        plt.text(0.95, 0.95, "%0.2f m" % hand_dist[frame], fontsize=12, color="black",
         ha="right", va="top", transform=plt.gca().transAxes)

        # Customize the plot
        plt.xlabel("frame")
        plt.ylabel("hand distance (m)")
        plt.title("Distance between Fc1 and Fc2 hands")
        plt.legend()

        # Save plot to image
        plot_filename = f"hand_dist_plot_{frame:04d}.png"
        plt.savefig(plot_filename)

        # Read plot image
        plot_image = cv2.imread(plot_filename)
        plot_image = cv2.resize(plot_image, size)

        # Write combined frame to output video
        out.write(plot_image)

        # Remove the plot image file
        os.remove(plot_filename)
        
        plt.close()

    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    #################### Using the rotation matrix of the session ####################
    rotation_matrix = np.array([[1.0248006017271942e-01, 1.5928201719510268e-01,
           -9.8189972821325022e-01], [-1.1369102978260752e-01,
           9.8250390051985037e-01, 1.4751418647116760e-01],
           [9.8821667007492131e-01, 9.6515928537931867e-02,
           1.1879599540596802e-01]])

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

    index_on_FC1 = 0
    index_on_FC2 = 0
    
    hand_dist = np.linalg.norm(landmarks_FC1[:, index_on_FC1] - landmarks_FC2[:, index_on_FC2], axis = 1)

    convert_to_video(hand_dist, './outputs/hand_dist.mp4')
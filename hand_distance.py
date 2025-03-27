import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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

    # Create a figure once with size matching the video dimensions
    dpi = 100
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    canvas = FigureCanvas(fig)  

    x = np.arange(hand_dist.shape[0])

    for frame in range(FRAME_COUNT):

        ax.cla()

        # Create the plot
        ax.plot(x, hand_dist, linestyle='-', color='b', label="Data")

        ax.scatter(x[frame], hand_dist[frame], color='red', s=100, label="Point")

        # Add text to the upper right corner
        ax.text(0.95, 0.95, "%0.2f m" % hand_dist[frame], fontsize=12, color="black",
         ha="right", va="top", transform=plt.gca().transAxes)

        # Customize the plot
        ax.set_xlabel("frame")
        ax.set_ylabel("hand distance (m)")
        ax.set_title("Distance between Fc1 and Fc2 hands")
        ax.legend()

        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

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
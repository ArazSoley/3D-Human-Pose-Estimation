import cv2
import mediapipe as mp
import numpy as np
from utils import *
import argparse
from scipy import stats

def get_2d_3d_pose(input_video_path, frame_count):
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,  # For video
        model_complexity=2,       # Highest accuracy (3D requires complexity=2)
        min_detection_confidence=0.7,
        min_tracking_confidence=0.95,
        smooth_landmarks = True
    )

    landmark_3d_arr = np.empty((frame_count, 33, 3))
    landmark_2d_arr = np.empty((frame_count, 33, 2))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = pose.process(frame_rgb)

        if results.pose_landmarks:

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

            landmark_3d_arr[frame_id] = landmarks_3d
            landmark_2d_arr[frame_id] = landmarks_2d

        frame_id += 1

        if frame_id == frame_count:
            break

    print(f"Processed {frame_id} frames.")

    # Release resources
    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    return landmark_3d_arr, landmark_2d_arr

def get_transformation_from_local_to_camera(landmark_3d_arr, landmark_2d_arr, fx, fy, cx, cy, dist_coeffs, threshold):
    
    t_arr = np.zeros((landmark_3d_arr.shape[0], 3))
    R_arr = np.zeros((landmark_3d_arr.shape[0], 3, 3))

    for frame in range(landmark_3d_arr.shape[0]):
        R_est, t_est = local_to_camera_transformation(landmark_3d_arr[frame], landmark_2d_arr[frame], fx, fy, cx, cy, dist_coeffs)

        landmark_3d_cam = transform_to_camera_frame(landmark_3d_arr[frame], R_est, t_est)

        if landmark_3d_cam[9][2] < 0:
            print("NEGATIVE")
            t_arr[frame] = t_arr[frame - 1]
            R_arr[frame] = R_arr[frame - 1]
        elif frame > 0 and np.linalg.norm(t_est.reshape(3,) - t_arr[frame - 1]) > threshold:
            print("THRESHOLD HIT")
            t_arr[frame] = t_arr[frame - 1]
            R_arr[frame] = R_arr[frame - 1]
        else:
            t_arr[frame] = t_est.reshape(3,)
            R_arr[frame] = R_est
        
    print("mean t:", np.mean(t_arr, axis=0))
    print("std t:", np.std(t_arr, axis=0))

    return t_arr, R_arr

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--FC', type=str, help='FC1 or FC2')
    parser.add_argument('--frame_count', type=int, help='Number of frames to process')
    
    args = parser.parse_args()

    FC = args.FC
    MAX_FRAME = args.frame_count
    fx = 1.71847e+03 if FC == "FC1" else 1.85417e+03
    fy = 1.53787e+03 if FC == "FC1" else 1.65964e+03
    cx = 6.23459e+02 if FC == "FC1" else 6.22157e+02
    cy = 3.76958e+02 if FC == "FC1" else 3.84014e+02
    dist_coeffs = np.array([-3.7094196768352616e-01, 1.8626642173380867e-01, 0., 0., 0.]) if FC == "FC1" else \
                np.array([-3.6485240781354333e-01, 1.9904269142924666e-01, 0., 0., 0.])

    # Input/Output paths
    input_video_path = f"../input_videos/{FC}_rtmw.mp4"

    landmark_3d_arr, landmark_2d_arr = get_2d_3d_pose(input_video_path, MAX_FRAME)

    # Finding transformation from local to camera frame
    t_arr, R_arr = get_transformation_from_local_to_camera(landmark_3d_arr, landmark_2d_arr, fx, fy, cx, cy, dist_coeffs, threshold=0.5)

    t_arr[:] = t_arr.mean(axis=0)

    # Transforming landmarks from local to camera frame
    landmark_3d_cam_arr = vectorized_transform_to_camera_frame(landmark_3d_arr, R_arr, t_arr)

    # Save landmark_3d_cam_arr to a .npy file
    np.save(f'./arrays/landmark_3d_cam_{FC}.npy', landmark_3d_cam_arr)

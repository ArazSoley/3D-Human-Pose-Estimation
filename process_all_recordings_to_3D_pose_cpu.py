import cv2
import mediapipe as mp
import numpy as np
from utils import *
import os
from tqdm import tqdm
import time

log_file = open("log_cpu.txt", "a")

def get_2d_3d_pose(input_video_path):
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the frame count of the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,  # For video
        model_complexity=2,       # Highest accuracy (3D requires complexity=2)
        min_detection_confidence=0.8,
        min_tracking_confidence=0.95,
        smooth_landmarks = True
    )

    landmark_3d_arr = np.empty((frame_count, 33, 3))
    landmark_2d_arr = np.empty((frame_count, 33, 2))

    for frame_id in tqdm(range(frame_count)):
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            results = pose.process(frame_rgb)

            if not results.pose_landmarks:
                print(f"No pose detected in frame {frame_id}.", file=log_file, flush=True)
                if frame_id == 0:
                    landmark_3d_arr[frame_id] = np.zeros((33, 3))
                    landmark_2d_arr[frame_id] = np.zeros((33, 2))
                else:
                    landmark_3d_arr[frame_id] = landmark_3d_arr[frame_id - 1]
                    landmark_2d_arr[frame_id] = landmark_2d_arr[frame_id - 1]
            else:
                # Extract 3D landmarks
                landmark_3d_arr[frame_id] = np.array(
                    [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_world_landmarks.landmark]
                )

                # Extract 2D landmarks
                landmark_2d_arr[frame_id] = np.array(
                    [[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark]
                )

    # Scale 2D landmarks
    landmark_2d_arr[:,:,0] *= width
    landmark_2d_arr[:,:,1] *= height

    print(f"Processed {frame_id} frames.", file=log_file, flush=True)

    # Release resources
    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    return landmark_3d_arr, landmark_2d_arr

def get_transformation_from_local_to_camera(landmark_3d_arr, landmark_2d_arr, fx, fy, cx, cy, dist_coeffs, threshold):
    
    t_arr = np.zeros((landmark_3d_arr.shape[0], 3))
    R_arr = np.zeros((landmark_3d_arr.shape[0], 3, 3))

    flag = True
    tmp_indices = []
    negatives_count = 0
    threshold_hit_count = 0

    for frame in range(landmark_3d_arr.shape[0]):
        R_est, t_est = local_to_camera_transformation(landmark_3d_arr[frame], landmark_2d_arr[frame], fx, fy, cx, cy, dist_coeffs)

        landmark_3d_cam = transform_to_camera_frame(landmark_3d_arr[frame], R_est, t_est)
        
        if landmark_3d_cam[9][2] < 1.0 or landmark_3d_cam[9][2] > 3.5:
            negatives_count += 1
            if flag:
                tmp_indices.append(frame)
            else:
                t_arr[frame] = t_arr[frame - 1]
                R_arr[frame] = R_arr[frame - 1]
        elif frame > 0 and not flag and np.linalg.norm(t_est.reshape(3,) - t_arr[frame - 1]) > threshold:
            threshold_hit_count += 1
            t_arr[frame] = t_arr[frame - 1]
            R_arr[frame] = R_arr[frame - 1]
        else:
            t_arr[frame] = t_est.reshape(3,)
            R_arr[frame] = R_est
            if flag:
                flag = False
                t_arr[tmp_indices] = t_arr[frame]
                R_arr[tmp_indices] = R_arr[frame]
    
    print("negatives count:", negatives_count, file=log_file, flush=True)
    print("threshold hit count:", threshold_hit_count, file=log_file, flush=True)

    print("mean t:", np.mean(t_arr, axis=0), file=log_file, flush=True)
    print("std t:", np.std(t_arr, axis=0), file=log_file, flush=True)

    return t_arr, R_arr

if __name__ == "__main__":
    dataset_path = "./UDIVAv0.5/recordings/lego_train"

    FC1_fx = 1716.43
    FC1_fy = 1536.58
    FC1_cx = 620.87
    FC1_cy = 375.52

    FC2_fx = 1734.83
    FC2_fy = 1552.56
    FC2_cx = 649.76
    FC2_cy = 388.29

    FC1_dist_coeffs = np.array([-0.371648, 0.1808984, 0.0, 0.0, 0.0])
    FC2_dist_coeffs = np.array([-0.37166988, 0.18559952, 0.0, 0.0, 0.0])

    # Iterate through all entries in the dataset directory
    for entry in os.listdir(dataset_path):
        
        # Construct the output directory
        out_dir = "./UDIVAv0.5/MediaPipe/lego_train/" + entry
        
        # Proceed if landmarks have not been processed yet
        if not os.path.exists(out_dir):
            # Construct the full path to the entry
            entry_path = os.path.join(dataset_path, entry)
            # Check if the entry is a directory
            if os.path.isdir(entry_path):

                print("processing:", entry)
                print("processing:", entry, file=log_file, flush=True)


                FC1_path = os.path.join(entry_path, "FC1_L.mp4")
                FC2_path = os.path.join(entry_path, "FC2_L.mp4")

                # Get 2D and 3D landmarks
                print(f"processing {entry} FC1")
                FC1_landmark_3d_arr, FC1_landmark_2d_arr = get_2d_3d_pose(FC1_path)
                print(f"processing {entry} FC2")
                FC2_landmark_3d_arr, FC2_landmark_2d_arr = get_2d_3d_pose(FC2_path)

                # Get transformation from local to camera
                FC1_t_arr, FC1_R_arr = get_transformation_from_local_to_camera(FC1_landmark_3d_arr, FC1_landmark_2d_arr, FC1_fx, FC1_fy, FC1_cx, FC1_cy, FC1_dist_coeffs, threshold=0.5)
                FC2_t_arr, FC2_R_arr = get_transformation_from_local_to_camera(FC2_landmark_3d_arr, FC2_landmark_2d_arr, FC2_fx, FC2_fy, FC2_cx, FC2_cy, FC2_dist_coeffs, threshold=0.5)

                FC1_t_arr[:] = FC1_t_arr.mean(axis=0)
                FC2_t_arr[:] = FC2_t_arr.mean(axis=0)

                # Transforming landmarks from local to camera frame
                FC1_landmark_3d_cam = vectorized_transform_to_camera_frame(FC1_landmark_3d_arr, FC1_R_arr, FC1_t_arr)
                FC2_landmark_3d_cam = vectorized_transform_to_camera_frame(FC2_landmark_3d_arr, FC2_R_arr, FC2_t_arr)

                # Create directory if it doesn't exist
                os.makedirs(out_dir, exist_ok=True)

                # Saving landmarks to file
                np.save(out_dir + "/FC1_landmark_3d_cam_cpu.npy", FC1_landmark_3d_cam)
                np.save(out_dir + "/FC2_landmark_3d_cam_cpu.npy", FC2_landmark_3d_cam)

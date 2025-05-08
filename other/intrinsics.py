import numpy as np
import cv2 as cv
import sys
import os


directory_path = "../../Calibration/calibration_files"

# fx = np.zeros((len(os.listdir(directory_path))))
# fy = np.zeros((len(os.listdir(directory_path))))
# cx = np.zeros((len(os.listdir(directory_path))))
# cy = np.zeros((len(os.listdir(directory_path))))

angles_array = np.zeros((len(os.listdir(directory_path)), 1))

# Iterate through all entries in the directory
for i, entry in enumerate(os.listdir(directory_path)):
    # Construct the full path
    full_path = os.path.join(directory_path, entry)
    
    # Check if the entry is a directory
    if os.path.isdir(full_path):
        file_name = os.path.join(full_path, entry[2:] + "_Z1-Z2.yml")

        s = cv.FileStorage(file_name, cv.FileStorage_READ)

        R = s.getNode('Rs').mat()
        eig, v = np.linalg.eig(R)

        rot_ax = v[:, 2].astype(np.float32)      
        angle = np.arctan2(eig[0].imag, eig[0].real)  

        angles_array[i] = angle

        angle2 = np.arccos((R[0][0] + R[1][1] + R[2][2]- 1) / 2)
        print(f"angle1: {angle}, angle2: {angle2}")
        # K = s.getNode('K0').mat()

        # fx[i] = K[0][0]
        # fy[i] = K[1][1]
        # cx[i] = K[0][2]
        # cy[i] = K[1][2]

        s.release()

# print("fx mean = ", np.mean(fx))
# print("fx std = ", np.std(fx))

# print("fy mean = ", np.mean(fy))
# print("fy std = ", np.std(fy))

# print("cx mean = ", np.mean(cx))
# print("cx std = ", np.std(cx))

# print("cy mean = ", np.mean(cy))
# print("cy std = ", np.std(cy))


# filename1 = "calib/calib_file/20190704/190704_Z1-Z2.yml"
# filename2 = "calib/calib_file/20190718/190718_Z1-Z2.yml"


# s1 = cv.FileStorage(filename1, cv.FileStorage_READ)
# s2 = cv.FileStorage(filename2, cv.FileStorage_READ)

# R1 = np.eye(3,3)
# T1 = np.zeros((3,1))

# R1 = s1.getNode('Rs').mat()
# T1 = s1.getNode('Ts').mat()

# R2 = s2.getNode('Rs').mat()
# T2 = s2.getNode('Ts').mat()


# print("R1 = \n", R1)
# print("T1 = \n", T1)

# print("R2 = \n", R2)
# print("T2 = \n", T2)

# eig1, v1 = np.linalg.eig(R1)
# eig2, v2 = np.linalg.eig(R2)

# print("eig1 = \n", eig1)
# print("eig2 = \n", eig2)

# print("v1 = \n", v1[0][0], v1[0][1], v1[0][2], "\n", v1[1][0], v1[1][1], v1[1][2], "\n", v1[2][0], v1[2][1], v1[2][2])

# print("v2 = \n", v2[0][0], v2[0][1], v2[0][2], "\n", v2[1][0], v2[1][1], v2[1][2], "\n", v2[2][0], v2[2][1], v2[2][2])

# rot_ax1 = v1[:, 2].astype(np.float32)
# rot_ax2 = v2[:, 2].astype(np.float32)

# print("angle between rot_ax1 and rot_ax2 = ", np.arccos(np.dot(rot_ax1, rot_ax2)) * 180 / np.pi, " degrees")

# angle1 = np.arctan2(eig1[0].imag, eig1[0].real)
# angle2 = np.arctan2(eig2[0].imag, eig2[0].real)

# print("angle1 = ", angle1 * 180 / np.pi, " degrees")
# print("angle2 = ", angle2 * 180 / np.pi, " degrees")


# s1.release()
# s2.release()
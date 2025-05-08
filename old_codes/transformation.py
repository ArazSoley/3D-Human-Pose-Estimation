import numpy as np
import matplotlib.pyplot as plt

def solve_projection_matrix(pts3d, pts2d):
    """
    Solve for projection matrix P using Direct Linear Transform (DLT)
    Args:
        pts3d: (N, 3) array of 3D joint coordinates (in local coordinate system)
        pts2d: (N, 2) array of corresponding 2D image coordinates
    Returns:
        P: (3, 4) projection matrix
    """
    assert len(pts3d) == len(pts2d), "Number of 2D and 3D points must match"
    assert len(pts3d) >= 6, "Need at least 6 points for DLT"

    # Build matrix A for homogeneous system Ap = 0
    A = []
    for i in range(len(pts3d)):
        X, Y, Z = pts3d[i]
        u, v = pts2d[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    
    A = np.array(A)
    
    # Solve using SVD (last column of V)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    P /= P[2,3]  # Normalize
    
    return P

def decompose_projection_matrix(P):
    """
    Decompose projection matrix P into K, R, and t using RQ decomposition
    Returns:
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix
        t: (3, ) translation vector
    """
    # Extract left 3x3 submatrix and perform RQ decomposition
    M = P[:, :3]
    
    # Use QR decomposition for RQ (reverse order using permutation matrix)
    H = np.eye(3)[::-1]
    Q, R = np.linalg.qr(H @ M.T @ H)
    
    # Recover K and R
    K = H @ R.T @ H
    R = H @ Q.T @ H
    
    # Ensure positive diagonal for K
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    
    # Normalize K (make K[2,2] = 1)
    K /= K[2,2]
    
    # Solve for translation vector t = K^-1 * P[:,3]
    t = np.linalg.inv(K) @ P[:,3]
    
    # Ensure proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        R *= -1
        t *= -1
    
    return K, R, t

def decompose_projection_matrix_with_fixed_intrinsics(P, fx, fy, cx, cy):
    """
    Decompose P into R and t, assuming fixed intrinsics K.
    Args:
        P: (3, 4) projection matrix
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point in pixels
    Returns:
        R: (3, 3) rotation matrix
        t: (3, ) translation vector
    """
    # Construct fixed intrinsic matrix K
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # Solve for extrinsics [R | t] = K^-1 P
    K_inv = np.linalg.inv(K)
    Rt = K_inv @ P
    
    # Extract R and t
    R = Rt[:, :3]
    t = Rt[:, 3]
    
    # Ensure R is a valid rotation matrix (orthogonal with det(R) = 1)
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R *= -1
        t *= -1
    
    return R, t

def transform_to_camera_frame(pts3d_local, R, t):
    """
    Transform local 3D points to camera coordinate system
    Args:
        pts3d_local: (N, 3) array of local 3D points
        R: (3, 3) rotation matrix
        t: (3, ) translation vector
    Returns:
        pts3d_cam: (N, 3) points in camera coordinates
    """
    return (R @ pts3d_local.T + t.reshape(-1, 1)).T

if __name__ == "__main__":

    # Load landmarks from file
    pts3d_local = np.load("landmarks.npz")["landmarks_3d"]
    pts2d = np.load("landmarks.npz")["landmarks_2d"]
    
    

    # Solve for projection matrix and decompose
    P = solve_projection_matrix(pts3d_local, pts2d)
    K_est, R_est, t_est = decompose_projection_matrix(P)
    
    # Transform points to camera coordinates
    pts3d_cam_est = transform_to_camera_frame(pts3d_local, R_est, t_est)
    
    # Plot 3D landmarks
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
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
            [pts3d_cam_est[start_idx, 0], pts3d_cam_est[end_idx, 0]],
            [pts3d_cam_est[start_idx, 2], pts3d_cam_est[end_idx, 2]],
            [-pts3d_cam_est[start_idx, 1], -pts3d_cam_est[end_idx, 1]],  # Note: Using z for y-axis
            c = 'b'
        )
    # Plot joints
    ax.scatter(
        pts3d_cam_est[:, 0],
        pts3d_cam_est[:, 1],
        pts3d_cam_est[:, 2],
        c='b',
        s=20  # Size of the joints
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    # ax.view_init(-90, -90, 0)  # Adjust camera angle for better visualization

    # Save plot to image
    plot_filename = f"plot.png"
    plt.savefig(plot_filename)
    plt.close(fig)

    

    # Print results
    print("Estimated Intrinsic Matrix:\n", K_est)
    print("\nEstimated Rotation Matrix:\n", R_est)
    print("\nEstimated Translation Vector:\n", t_est)

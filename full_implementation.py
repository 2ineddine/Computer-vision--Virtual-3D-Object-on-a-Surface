import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to manually compute the homography matrix
def compute_homography(src_points, dst_points):
    A = []
    for i in range(len(src_points)):
        x, y = src_points[i]
        u, v = dst_points[i]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A)

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))
    return H / H[2, 2]

# Compute 3D to 2D projection matrix from homography
def compute_projection_matrix(obj_points, img_points, mtx):
    H = compute_homography(obj_points[:, :2], img_points)
    inv_K = np.linalg.inv(mtx)

    # Decompose homography matrix to extract rotation and translation
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    lambda_ = 1 / np.linalg.norm(np.dot(inv_K, h1))
    r1 = lambda_ * np.dot(inv_K, h1)
    r2 = lambda_ * np.dot(inv_K, h2)
    r3 = np.cross(r1, r2)
    t = lambda_ * np.dot(inv_K, h3)
    R = np.column_stack([r1, r2, r3])

    # Combine rotation and translation
    RT = np.column_stack((R, t))

    # Compute final projection matrix
    P = mtx @ RT
    return P

# Chessboard configuration
chessboard_size = (9, 6)
square_size = 25  # in mm or pixels depending on your calibration

# 3D points of the chessboard pattern (object points)
obj_p = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_p[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
obj_p *= square_size

# Get the current script directory and build the video path
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "vid3.mp4")

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"The video file '{video_path}' does not exist.")
    exit()

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error opening the video.")
    exit()

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Cannot read the video.")
    cap.release()
    exit()

# Convert to grayscale
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Detect chessboard corners in the first frame
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
if not ret:
    print("Chessboard not detected in the first frame.")
    cap.release()
    exit()

# Camera calibration
obj_points = [obj_p]          # 3D coordinates of chessboard corners
img_points = [corners]        # Corresponding 2D image points
ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Define the 3D cube (8 corners of the cube)
scale_factor = 50
cube_3d = np.array([
    [-0.5, -0.5, 0],
    [0.5, -0.5, 0],
    [0.5, 0.5, 0],
    [-0.5, 0.5, 0],
    [-0.5, -0.5, -1],
    [0.5, -0.5, -1],
    [0.5, 0.5, -1],
    [-0.5, 0.5, -1]
], dtype=np.float32) * scale_factor

# Manual translation offset for cube alignment
offset = np.array([0, 0, 0])

resize_factor = 1  # Resize factor for the displayed video

# Main loop to process each video frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or read error.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        corners = corners.reshape(-1, 2)

        # Compute projection matrix
        P = compute_projection_matrix(obj_p, corners, mtx)

        # Adjust the cube position based on the chessboard reference
        cube_3d_translated = cube_3d + np.array([obj_p[0, 0], obj_p[0, 1], 0]) + offset

        # Project the 3D points to 2D using the projection matrix
        cube_3d_homo = np.hstack((cube_3d_translated, np.ones((cube_3d_translated.shape[0], 1))))
        projected_points = P @ cube_3d_homo.T
        projected_points /= projected_points[2, :]
        projected_points = projected_points[:2, :].T

        # Draw the edges of the cube
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), 
                 (4, 5), (5, 6), (6, 7), (7, 4), 
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for edge in edges:
            pt1 = tuple(np.int32(projected_points[edge[0], :2]))
            pt2 = tuple(np.int32(projected_points[edge[1], :2]))
            if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    # Display the current frame
    resized_frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    cv2.imshow("Frame Preview", resized_frame)

    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Video manually stopped by the user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

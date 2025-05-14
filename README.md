## Project: 3D Cube Projection on Checkerboard using OpenCV

This project uses computer vision techniques to project a 3D virtual cube onto a checkerboard detected in a video. The cube remains fixed relative to the board even when the camera moves. The process relies on **camera calibration** to determine the intrinsic camera matrix. After calibration, the **homography** between the 3D model points and their 2D image projections is computed to define the transformation needed to project a 3D object onto a 2D image plane.

The code uses **OpenCV** and **NumPy**:

- OpenCV is used to detect checkerboard corners, calibrate the camera, and draw cube edges.
- NumPy is used for matrix operations related to homography and projection.

The result is a video where the virtual cube appears fixed on the checkerboard, unaffected by camera motion.

### Methods

#### 1. Camera Calibration

Calibration estimates the camera's **intrinsic** and **extrinsic** parameters to relate 3D real-world coordinates to 2D image coordinates.

##### Intrinsic Matrix \(\mathbf{K}\)

The intrinsic matrix \(\mathbf{K}\) is:

```math
\mathbf{K} = \begin{bmatrix}
    f_x & s & c_x \\
    0 & f_y & c_y \\
    0 & 0 & 1
\end{bmatrix}
```

- \(f_x, f_y\): focal lengths in pixels
- \(s\): skew coefficient (usually 0)
- \(c_x, c_y\): coordinates of the principal point

Calibration can be performed using MATLAB or Python. In our project, we use OpenCV functions like `cv2.findChessboardCorners` and `cv2.calibrateCamera`.

#### 2. Homography Estimation

Homography maps points from one plane to another. The 2D image point \([u, v, 1]^T\) is related to 3D model coordinates \([X, Y, 1]^T\) via:

```math
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = H \cdot \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
```

Where:

```math
H = K \cdot \begin{bmatrix} R & T \end{bmatrix}
```

Or:

```math
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \cdot \begin{bmatrix} R & T \end{bmatrix} \cdot \begin{bmatrix} X \\ Y \\ 0 \\ 1 \end{bmatrix}
```

Or:

```math
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} r_{11} & r_{12} & t_x \\ r_{21} & r_{22} & t_y \\ r_{31} & r_{32} & t_z \end{bmatrix} \cdot \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
```

We solve:

```math
A \cdot h = 0
```

Using **SVD decomposition**.

#### 3. Camera Pose Estimation

Given the homography \(H\) and intrinsic matrix \(K\), we decompose \(H\) into rotation \(R\) and translation \(T\):

```math
H = \begin{bmatrix} h_1 & h_2 & h_3 \end{bmatrix}
```

```math
K^{-1} \cdot H = [r_1\ r_2\ t]
```

```math
\lambda = \frac{1}{\|K^{-1} h_1\|}
```

```math
r_1 = \lambda K^{-1} h_1, \quad r_2 = \lambda K^{-1} h_2
```

```math
r_3 = r_1 \times r_2
```

```math
t = \lambda K^{-1} h_3
```

```math
R = \begin{bmatrix} r_1 & r_2 & r_3 \end{bmatrix}
```

```math
P = K \cdot [R | T]
```

#### 4. Primitive Tracking

Steps:

1. Load the video and extract first frame.
2. Detect checkerboard corners using `cv2.findChessboardCorners()`.
3. Estimate camera matrix using `cv2.calibrateCamera()`.
4. Estimate homography.
5. Decompose homography into \(R\) and \(T\).
6. Build a 3D cube model.
7. Project 3D cube onto 2D image using projection matrix \(P\).
8. Draw cube using `cv2.line()`.

---

### Dependencies

- OpenCV
- NumPy

### Run Instructions

```bash
full_implementation.py
```


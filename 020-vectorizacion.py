import cv2
import numpy as np
import random
import time
import math  # for sin, cos
import sys
from numba import njit, prange

# ---------------------
# CAMERA ROTATION HELPERS
# ---------------------
@njit
def rotation_matrix_numba(theta_x, theta_y, theta_z):
    """Combined rotation matrix for X, Y, Z axes using Numba."""
    cosx = math.cos(theta_x)
    sinx = math.sin(theta_x)
    cosy = math.cos(theta_y)
    siny = math.sin(theta_y)
    cosz = math.cos(theta_z)
    sinz = math.sin(theta_z)
    
    # Rotation matrices
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cosx, -sinx],
        [0.0, sinx, cosx]
    ], dtype=np.float32)
    
    Ry = np.array([
        [cosy, 0.0, siny],
        [0.0, 1.0, 0.0],
        [-siny, 0.0, cosy]
    ], dtype=np.float32)
    
    Rz = np.array([
        [cosz, -sinz, 0.0],
        [sinz, cosz, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    # Combined rotation
    return Rz @ Ry @ Rx

# ---------------------
# PERSPECTIVE PROJECTION
# ---------------------
@njit
def project_numba(vertices, center_factor, perspective=600):
    """
    Vectorized perspective projection with Numba.
    vertices: (N, 3)
    center_factor: (2,)
    """
    N = vertices.shape[0]
    projected = np.empty((N, 2), dtype=np.int32)
    for i in range(N):
        y = vertices[i, 1] + perspective
        if y < 1e-3:
            y = 1e-3
        factor = perspective / y
        x = vertices[i, 0] * factor + center_factor[0]
        z = vertices[i, 2] * factor + center_factor[1]
        projected[i, 0] = int(x)
        projected[i, 1] = int(z)
    return projected

# ---------------------
# GENERATE SHAPES
# ---------------------
def generate_shapes(num_cubes, height_limit=4200):
    """
    Generates positions for all cubes.
    """
    centers = np.random.uniform(-4000, 4000, size=(num_cubes, 3)).astype(np.float32)
    centers[:, 1] = np.random.uniform(0, height_limit, size=num_cubes)  # y-coordinate
    dimensions = np.array([0.2, 1, 0.2], dtype=np.float32)  # Same for all cubes
    colors = np.array([255, 255, 255], dtype=np.uint8)  # White color for all
    return centers, dimensions, colors

# ---------------------
# MAIN FUNCTION
# ---------------------
def main():
    # Constants
    WIDTH, HEIGHT = 1920, 1080
    FPS = 30
    DURATION = 60 * 60  # 1-hour video at 30 FPS
    NUM_CUBES = random.randint(6000, 12000)
    OUTPUT_FILE = f'starfield_screensaver_{round(time.time())}.mp4'
    PERSPECTIVE = 600
    CENTER_FACTOR = np.array([WIDTH // 2, HEIGHT // 2], dtype=np.float32)
    
    # Generate shapes
    print(f"Generating {NUM_CUBES} cubes...")
    centers, dimensions, colors = generate_shapes(NUM_CUBES)
    print("Generation complete.")
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))
    
    # Initialize frame buffer
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    frames = int(FPS * DURATION)
    start_time = time.time()
    
    # Warm up Numba (optional but can avoid first-call delay)
    rotation_matrix_numba(0.0, 0.0, 0.0)
    project_numba(np.zeros((1,3), dtype=np.float32), CENTER_FACTOR)
    
    print("Starting simulation...")
    for frame_idx in range(frames):
        # 1) Apply semi-transparent black overlay (motion blur)
        alpha = 0.6
        cv2.addWeighted(frame, alpha, frame, 1 - alpha, 0, frame)
        
        # 2) Compute camera rotation
        angle_x = 0.2 * math.sin(frame_idx * 0.01)
        angle_y = 0.15 * math.cos(frame_idx * 0.007)
        angle_z = 0.1 * math.sin(frame_idx * 0.005)
        R = rotation_matrix_numba(angle_x, angle_y, angle_z)
        
        # 3) Rotate all centers
        rotated_centers = centers @ R.T  # Shape: (NUM_CUBES, 3)
        
        # 4) Project all centers
        projected = project_numba(rotated_centers, CENTER_FACTOR, PERSPECTIVE)  # Shape: (NUM_CUBES, 2)
        
        # 5) Sort shapes by depth (y-coordinate)
        sorted_indices = np.argsort(-rotated_centers[:, 1])  # Descending order
        rotated_centers_sorted = rotated_centers[sorted_indices]
        projected_sorted = projected[sorted_indices]
        colors_sorted = colors[sorted_indices]
        
        # 6) Create temporary buffer
        temp_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # 7) Draw points (simplified rendering)
        # Vectorized drawing using NumPy's advanced indexing
        valid = (projected_sorted[:, 0] >= 0) & (projected_sorted[:, 0] < WIDTH) & \
                (projected_sorted[:, 1] >= 0) & (projected_sorted[:, 1] < HEIGHT)
        valid_x = projected_sorted[valid, 0]
        valid_y = projected_sorted[valid, 1]
        temp_buffer[valid_y, valid_x] = colors_sorted[valid]
        
        # 8) Update frame with temp_buffer
        frame = cv2.add(frame, temp_buffer)
        
        # 9) Update cube positions (falling effect)
        centers[:, 1] -= 3  # Move down
        reset_mask = centers[:, 1] < -HEIGHT // 2
        num_resets = np.sum(reset_mask)
        if num_resets > 0:
            centers[reset_mask, 1] = np.random.uniform(2800, 4200, size=num_resets)
        
        # 10) Write frame to video
        out.write(frame)
        
        # Print stats every 30 frames
        if frame_idx % 30 == 0:
            elapsed_time = time.time() - start_time
            progress = (frame_idx / frames) * 100
            if frame_idx > 0:
                fps_est = frame_idx / elapsed_time
                time_remaining = (frames - frame_idx) / fps_est
            else:
                time_remaining = 0
            print(f"Frame {frame_idx}/{frames} ({progress:.2f}%): "
                  f"Elapsed={elapsed_time:.2f}s, ETA={time_remaining:.2f}s")
        
        # 11) (Optional) Display the framebuffer
        cv2.imshow('Framebuffer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting early...")
            break
    
    # Release everything
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

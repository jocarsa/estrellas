import cv2
import numpy as np
import random
import time
import math  # for sin, cos
import sys

# For parallel computations
from joblib import Parallel, delayed

# Initialize CENTER_FACTOR as None
CENTER_FACTOR = None

# ---------------------
# CAMERA ROTATION HELPERS
# ---------------------
def rotation_x(theta):
    """Rotation matrix around X-axis by angle theta."""
    return np.array([
        [1,           0,            0],
        [0,  math.cos(theta), -math.sin(theta)],
        [0,  math.sin(theta),  math.cos(theta)]
    ], dtype=np.float32)

def rotation_y(theta):
    """Rotation matrix around Y-axis by angle theta."""
    return np.array([
        [ math.cos(theta), 0, math.sin(theta)],
        [              0, 1,             0],
        [-math.sin(theta), 0, math.cos(theta)]
    ], dtype=np.float32)

def rotation_z(theta):
    """Rotation matrix around Z-axis by angle theta."""
    return np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [             0,               0, 1]
    ], dtype=np.float32)

# Build a combined rotation matrix for the camera
def get_camera_rotation(frame_idx):
    """
    Let the camera angles oscillate with time.
    Adjust frequencies (0.01, 0.007, 0.005) and
    amplitudes (0.2, 0.15, 0.1) to taste.
    """
    angle_x = 0.2 * math.sin(frame_idx * 0.01)
    angle_y = 0.15 * math.cos(frame_idx * 0.007)
    angle_z = 0.1 * math.sin(frame_idx * 0.005)

    Rx = rotation_x(angle_x)
    Ry = rotation_y(angle_y)
    Rz = rotation_z(angle_z)

    # Order of multiplication determines how rotations combine
    return Rz @ Ry @ Rx

# ---------------------
# BOX CLASS
# ---------------------
class Box:
    def __init__(self, center, dimensions, color):
        self.center = np.array(center, dtype=np.float32)
        self.dimensions = np.array(dimensions, dtype=np.float32)
        self.color = color

    def vertices(self):
        x, y, z = self.center
        dx, dy, dz = self.dimensions
        return np.array([
            [x - dx, y - dy, z - dz],
            [x + dx, y - dy, z - dz],
            [x + dx, y + dy, z - dz],
            [x - dx, y + dy, z - dz],
            [x - dx, y - dy, z + dz],
            [x + dx, y - dy, z + dz],
            [x + dx, y + dy, z + dz],
            [x - dx, y + dy, z + dz],
        ], dtype=np.float32)

# ---------------------
# PERSPECTIVE PROJECTION
# (Now expects you to pass already camera-rotated vertices)
# ---------------------
PERSPECTIVE = 600  # Perspective projection factor

def project(vertices, center_factor):
    """
    vertices: (N, 3) in camera space
    Returns:  (N, 2) integer screen coordinates
    We use y in the denominator => perspective factor
    x->screenX, z->screenY
    """
    y_values = vertices[:, 1] + PERSPECTIVE
    # Clip to avoid negative or zero division
    y_values = np.clip(y_values, 1e-3, None)
    factors = PERSPECTIVE / y_values

    # x->Xscreen, z->Yscreen
    projected_xz = vertices[:, [0, 2]] * factors[:, None]
    return (projected_xz + center_factor).astype(np.int32)

# ---------------------
# DRAW SHAPE
# ---------------------
def draw_shape(frame, shape, temp_buffer, camera_rotation, center_factor):
    """
    Draws a single shape onto the temp_buffer using the camera rotation.
    """
    # 1) get original vertices
    verts = shape.vertices()
    # 2) rotate them by the camera rotation
    rotated_verts = verts @ camera_rotation.T  # shape (8,3)
    # 3) project to 2D
    projected_vertices = project(rotated_verts, center_factor)

    # Define faces (indices + average depth)
    # Using y' as 'depth' in camera space
    faces = [
        ([0, 1, 2, 3], np.mean(rotated_verts[[0, 1, 2, 3], 1])),
        ([4, 5, 6, 7], np.mean(rotated_verts[[4, 5, 6, 7], 1])),
        ([0, 1, 5, 4], np.mean(rotated_verts[[0, 1, 5, 4], 1])),
        ([2, 3, 7, 6], np.mean(rotated_verts[[2, 3, 7, 6], 1])),
        ([1, 2, 6, 5], np.mean(rotated_verts[[1, 2, 6, 5], 1])),
        ([0, 3, 7, 4], np.mean(rotated_verts[[0, 3, 7, 4], 1])),
    ]

    # Sort faces by depth, descending
    faces.sort(key=lambda face: face[1], reverse=True)

    # Draw each face
    for face_indices, _ in faces:
        pts = np.array([projected_vertices[i] for i in face_indices], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillConvexPoly(temp_buffer, pts, shape.color, lineType=cv2.LINE_AA)

# ---------------------
# GENERATE SHAPES
# ---------------------
def generate_shapes(num_cubes):
    """
    Generates a list of Box objects with random positions.
    """
    return [
        Box(
            center=[
                random.uniform(-4000, 4000),
                random.uniform(0, 4200),
                random.uniform(-4000, 4000)
            ],
            dimensions=[0.2, 1, 0.2],
            color=(255, 255, 255)
        )
        for _ in range(num_cubes)
    ]

# ---------------------
# PARALLEL FACE EXTRACTION
# ---------------------
def get_faces_from_shape(shape, camera_rotation, center_factor):
    """
    Extracts all faces from a shape after applying camera rotation and projection.
    Returns a list of tuples: (projected_points, color, mean_depth)
    """
    verts = shape.vertices()
    rotated_verts = verts @ camera_rotation.T  # shape (8,3)
    projected_verts = project(rotated_verts, center_factor)

    # Each face is (indices, mean_depth)
    face_indices_list = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ]

    faces = []
    for f_indices in face_indices_list:
        # Gather the 2D points
        pts_2d = projected_verts[f_indices]
        # Average depth in camera space (rotated_verts)
        mean_depth = np.mean(rotated_verts[f_indices, 1])
        faces.append((pts_2d, shape.color, mean_depth))

    return faces

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    global CENTER_FACTOR  # Declare global to modify it

    for _ in range(1):  # Just run once for demonstration

        # Constants
        WIDTH, HEIGHT = 1920, 1080
        FPS = 30
        DURATION = 60 * 60  # 1-hour video at 30 FPS
        NUM_CUBES = random.randint(6000, 12000)
        OUTPUT_FILE = f'starfield_screensaver_{round(time.time())}.mp4'

        # Initialize CENTER_FACTOR after knowing WIDTH and HEIGHT
        CENTER_FACTOR = np.array([WIDTH // 2, HEIGHT // 2])

        # Generate shapes
        print(f"Generating {NUM_CUBES} cubes...")
        shapes = generate_shapes(NUM_CUBES)
        print("Generation complete.")

        # Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

        # Initialize frame buffer
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        frames = int(FPS * DURATION)
        start_time = time.time()

        print("Starting simulation...")
        for frame_idx in range(frames):
            # 1) Apply semi-transparent black overlay (motion blur)
            overlay = np.zeros_like(frame, dtype=np.uint8)
            alpha = 0.6
            cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0, frame)

            # 2) Create temporary buffer for new shapes
            temp_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            # 3) Sort shapes by some approximate depth (optional)
            shapes.sort(key=lambda s: s.center[1], reverse=True)

            # 4) Get camera rotation for this frame
            camera_rotation = get_camera_rotation(frame_idx)

            # 5a) PARALLEL step: gather faces for all shapes
            all_faces = Parallel(n_jobs=32)(
                delayed(get_faces_from_shape)(shape, camera_rotation, CENTER_FACTOR)
                for shape in shapes
            )
            # Flatten: all_faces is a list of lists
            # Each shape gave us 6 faces => flatten them
            face_data_list = [face for shape_faces in all_faces for face in shape_faces]

            # 5b) Sort all faces by mean_depth (descending)
            face_data_list.sort(key=lambda f: f[2], reverse=True)

            # 5c) Now draw them in the correct order (painter's algorithm).
            for (pts_2d, color, _) in face_data_list:
                pts_2d = pts_2d.reshape((-1, 1, 2))
                cv2.fillConvexPoly(temp_buffer, pts_2d, color, lineType=cv2.LINE_AA)

            # 5d) Update shape positions (falling effect) - done sequentially
            for shape in shapes:
                shape.center[1] -= 3  # tweak fall speed
                if shape.center[1] < -HEIGHT // 2:
                    shape.center[1] = random.uniform(2800, 4200)  # reset

            # 6) Apply glow effect
            blurred = cv2.GaussianBlur(temp_buffer, (21, 21), sigmaX=0, sigmaY=0)
            glow = cv2.addWeighted(temp_buffer, 1.0, blurred, 0.5, 0)
            cv2.add(frame, glow, frame)

            # 7) Write to video
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

            # 8) (Optional) Display the framebuffer
            #cv2.imshow('Framebuffer', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting early...")
                break

        # Release everything
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved to {OUTPUT_FILE}")

# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    main()

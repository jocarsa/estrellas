import cv2
import random
import time
import math  # for sin, cos
import sys

# ---------------------------------
# NEW: Use CuPy for GPU arrays
# ---------------------------------
import cupy as cp

# Constants
WIDTH, HEIGHT = 1920, 1080  
FPS = 30                    
DURATION = 60*60              # 1 hour for demonstration
NUM_CUBES = random.randint(6000, 12000)
OUTPUT_FILE = f'starfield_screensaver_{round(time.time())}.mp4'

# Perspective projection factor
PERSPECTIVE = 600

# We'll keep the CPU version of this for final offset in 2D:
CENTER_FACTOR = (WIDTH // 2, HEIGHT // 2)

# ---------------------
# CAMERA ROTATION HELPERS (GPU versions)
# ---------------------
def rotation_x(theta):
    """Rotation matrix around X-axis by angle theta (on GPU)."""
    return cp.array([
        [1,            0,             0],
        [0,  math.cos(theta), -math.sin(theta)],
        [0,  math.sin(theta),  math.cos(theta)]
    ], dtype=cp.float32)

def rotation_y(theta):
    """Rotation matrix around Y-axis by angle theta (on GPU)."""
    return cp.array([
        [ math.cos(theta), 0, math.sin(theta)],
        [              0, 1,              0],
        [-math.sin(theta), 0, math.cos(theta)]
    ], dtype=cp.float32)

def rotation_z(theta):
    """Rotation matrix around Z-axis by angle theta (on GPU)."""
    return cp.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta),  math.cos(theta), 0],
        [             0,               0, 1]
    ], dtype=cp.float32)

def get_camera_rotation(frame_idx):
    """
    Let the camera angles oscillate with time.
    Frequencies:  0.01, 0.007, 0.005
    Amplitudes:   0.2,  0.15,  0.1
    """
    angle_x = 0.2 * math.sin(frame_idx * 0.01)
    angle_y = 0.15 * math.cos(frame_idx * 0.007)
    angle_z = 0.1 * math.sin(frame_idx * 0.005)

    Rx = rotation_x(angle_x)
    Ry = rotation_y(angle_y)
    Rz = rotation_z(angle_z)

    # Matrix multiply on GPU
    return Rz @ Ry @ Rx  # shape (3,3)

# ---------------------
# BOX CLASS (GPU-friendly)
# ---------------------
class Box:
    def __init__(self, center, dimensions, color):
        # We'll store center/dimensions as GPU arrays
        self.center = cp.array(center, dtype=cp.float32)
        self.dimensions = cp.array(dimensions, dtype=cp.float32)
        self.color = color  # keep color as a tuple (B,G,R) for CPU drawing

    def vertices(self):
        """
        Return shape (8,3) array of the 8 corners, on GPU (cupy).
        """
        x, y, z = self.center
        dx, dy, dz = self.dimensions
        return cp.array([
            [x - dx, y - dy, z - dz],
            [x + dx, y - dy, z - dz],
            [x + dx, y + dy, z - dz],
            [x - dx, y + dy, z - dz],
            [x - dx, y - dy, z + dz],
            [x + dx, y - dy, z + dz],
            [x + dx, y + dy, z + dz],
            [x - dx, y + dy, z + dz],
        ], dtype=cp.float32)

# ---------------------
# PERSPECTIVE PROJECTION (GPU)
# ---------------------
def project(vertices):
    """
    vertices: (N,3) in camera-space (on GPU).
    We'll treat the second coordinate (vertices[:,1]) as 'depth'.
    We'll produce a 2D result (on GPU), then move to CPU.
    """
    # Add perspective offset
    y_values = vertices[:, 1] + PERSPECTIVE
    
    # Avoid division by zero or negative depths
    y_values = cp.clip(y_values, 1e-3, None)

    factors = PERSPECTIVE / y_values
    # We'll map x->Xscreen, z->Yscreen
    projected_xz = vertices[:, [0, 2]] * factors[:, None]
    return projected_xz  # shape (N,2) still on GPU

# ---------------------
# DRAW SHAPE (mix GPU + CPU)
# ---------------------
def draw_shape(frame_cpu, shape, temp_buffer, camera_rotation):
    """
    frame_cpu: (H,W,3) CPU image.
    shape: Box instance with GPU center/dimensions.
    temp_buffer: same shape as frame_cpu, CPU.
    camera_rotation: (3,3) GPU matrix.
    """
    # 1) get original vertices (GPU)
    verts_gpu = shape.vertices()
    
    # 2) rotate them by the camera rotation (GPU)
    rotated_verts_gpu = verts_gpu @ camera_rotation.T  # shape (8,3)

    # 3) project to 2D (GPU)
    projected_gpu = project(rotated_verts_gpu)  # shape (8,2) on GPU

    # 4) Move projected coords to CPU for drawing
    projected_cpu = cp.asnumpy(projected_gpu)

    # 4.1) Also need 'rotated_verts_gpu' for face-sorting by depth
    rotated_cpu = cp.asnumpy(rotated_verts_gpu)

    # Center them on screen in CPU
    projected_cpu[:, 0] += CENTER_FACTOR[0]  # X
    projected_cpu[:, 1] += CENTER_FACTOR[1]  # Y

    # Round or convert to int
    projected_cpu = projected_cpu.astype(int)

    # Define 6 faces with average depth
    faces = [
        ([0, 1, 2, 3],  rotated_cpu[[0, 1, 2, 3], 1].mean()),
        ([4, 5, 6, 7],  rotated_cpu[[4, 5, 6, 7], 1].mean()),
        ([0, 1, 5, 4],  rotated_cpu[[0, 1, 5, 4], 1].mean()),
        ([2, 3, 7, 6],  rotated_cpu[[2, 3, 7, 6], 1].mean()),
        ([1, 2, 6, 5],  rotated_cpu[[1, 2, 6, 5], 1].mean()),
        ([0, 3, 7, 4],  rotated_cpu[[0, 3, 7, 4], 1].mean()),
    ]
    # Sort faces by depth descending
    faces.sort(key=lambda f: f[1], reverse=True)

    # Draw each face on CPU
    for face_indices, _ in faces:
        pts = projected_cpu[face_indices]
        pts = pts.reshape((-1, 1, 2))
        cv2.fillConvexPoly(temp_buffer, pts, shape.color, lineType=cv2.LINE_AA)

# ---------------------
# GENERATE SHAPES (GPU centers)
# ---------------------
shapes = []
for _ in range(NUM_CUBES):
    center = [
        random.uniform(-4000, 4000),
        random.uniform(0, 4200),
        random.uniform(-4000, 4000)
    ]
    dimensions = [0.2, 1, 0.2]
    color = (255, 255, 255)  # white
    shapes.append(Box(center, dimensions, color))

# ---------------------
# VIDEO WRITER
# ---------------------
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

# Initialize frame buffer (CPU)
frame_cpu = cp.zeros((HEIGHT, WIDTH, 3), dtype=cp.uint8).get()

frames = int(FPS * DURATION)
start_time = time.time()

for frame_idx in range(frames):
    # 1) "Motion blur" overlay on CPU
    overlay_cpu = cp.zeros_like(cp.array(frame_cpu))
    alpha = 0.6
    # Weighted sum on CPU
    cv2.addWeighted(frame_cpu, alpha, overlay_cpu.get(), 1 - alpha, 0, frame_cpu)

    # 2) temp_buffer for new shapes (CPU)
    temp_buffer = cp.zeros_like(cp.array(frame_cpu)).get()

    # 3) Sort shapes by approximate depth. 
    #    Because shape.center is on GPU, let's get them as CPU floats.
    #    Or, do a CPU-based approximation by storing .center[1].get() each time.
    #    (You could keep a CPU cached copy of center[1].)
    shapes.sort(key=lambda s: float(s.center[1].get()), reverse=True)

    # 4) camera rotation (GPU)
    camera_rotation = get_camera_rotation(frame_idx)  # (3,3) on GPU

    # 5) Draw & update shapes
    for shape in shapes:
        draw_shape(frame_cpu, shape, temp_buffer, camera_rotation)

        # Update shape position (falling effect) - do on GPU
        # shape.center is a cupy array; we can do:
        shape.center[1] -= 3.0
        
        # If it's too far "behind" the camera, reset
        # (We do .item() to convert single-element GPU array to CPU float)
        if shape.center[1].item() < -HEIGHT // 2:
            shape.center[1] = cp.float32(random.uniform(2800, 4200))

    # 6) "Glow" effect on CPU
    blurred = cv2.GaussianBlur(temp_buffer, (21, 21), sigmaX=0, sigmaY=0)
    glow = cv2.addWeighted(temp_buffer, 1.0, blurred, 0.5, 0)
    cv2.add(frame_cpu, glow, frame_cpu)

    # 7) Write to video (CPU)
    out.write(frame_cpu)

    # Print stats every 30 frames
    if frame_idx % 30 == 0:
        elapsed_time = time.time() - start_time
        progress = (frame_idx / frames) * 100
        if frame_idx > 0:
            fps_measured = frame_idx / elapsed_time
            eta = (frames - frame_idx) / fps_measured
        else:
            eta = 0
        print(f"Frame {frame_idx}/{frames} ({progress:.2f}%): "
              f"Elapsed={elapsed_time:.2f}s, ETA={eta:.2f}s")

    # 8) (Optional) Display
    cv2.imshow('Framebuffer', frame_cpu)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
print(f"Video saved to {OUTPUT_FILE}")

import cv2
import numpy as np
import random
import time
import math  # for sin, cos
import sys
from noise import pnoise2  # Perlin noise function

for _ in range(0, 1):  # Just run once for demonstration

    # Constants
    WIDTH, HEIGHT = 1920, 1080  
    FPS = 30                    
    DURATION = 60*60               # 60-minute test
    NUM_CUBES = random.randint(600, 1200)  
    OUTPUT_FILE = f'starfield_screensaver_{round(time.time())}.mp4'

    # Perspective projection factor
    PERSPECTIVE = 600
    CENTER_FACTOR = np.array([WIDTH // 2, HEIGHT // 2])

    # ---------------------
    # CAMERA ROTATION HELPERS
    # ---------------------
    def rotation_x(theta):
        """Rotation matrix around X-axis by angle theta."""
        return np.array([
            [1,           0,            0],
            [0,  np.cos(theta), -np.sin(theta)],
            [0,  np.sin(theta),  np.cos(theta)]
        ], dtype=np.float32)

    def rotation_y(theta):
        """Rotation matrix around Y-axis by angle theta."""
        return np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [            0, 1,            0],
            [-np.sin(theta), 0, np.cos(theta)]
        ], dtype=np.float32)

    def rotation_z(theta):
        """Rotation matrix around Z-axis by angle theta."""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [           0,             0, 1]
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
        # e.g., R = Rz * Ry * Rx
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
    def project(vertices):
        # vertices: shape (N, 3) => already in "camera space"
        # perspective: scale = PERSPECTIVE / (y + PERSPECTIVE)
        # final: (x * scale, z * scale)
        # then center on (WIDTH//2, HEIGHT//2)

        # Avoid division by zero or negative depths
        y_values = vertices[:, 1] + PERSPECTIVE

        # For robust handling, clip small or negative values
        # (or you can skip if you assume everything is in front of camera)
        y_values = np.clip(y_values, 1e-3, None)

        factors = PERSPECTIVE / y_values
        # We'll map x->Xscreen, z->Yscreen in your scheme
        projected_xz = vertices[:, [0, 2]] * factors[:, None]
        return (projected_xz + CENTER_FACTOR).astype(np.int32)

    # ---------------------
    # DRAW SHAPE
    # ---------------------
    def draw_shape(frame, shape, temp_buffer, camera_rotation):
        # 1) get original vertices
        verts = shape.vertices()  
        # 2) rotate them by the camera rotation
        rotated_verts = verts @ camera_rotation.T  # shape (8,3)
        # 3) project to 2D
        projected_vertices = project(rotated_verts)

        # Define faces (indices + average depth)
        # Using y' as 'depth' in camera space, but you can pick z' if you prefer
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
    # GENERATE SHAPES WITH PERLIN NOISE
    # ---------------------
    from noise import pnoise2  # Ensure this import is present

    # Constants for Perlin noise
    PERLIN_SCALE = 0.002  # Adjusted to control clustering scale
    CLUSTER_OFFSET = 50  # Maximum offset to apply based on noise

    def generate_perlin_position():
        """
        Generates a position with uniform distribution, then perturbs it based on Perlin noise
        to create clustering effects.
        """
        # Uniformly distribute x and z
        x = random.uniform(-4000, 4000)
        z = random.uniform(-4000, 4000)
        y = random.uniform(0, 4200)  # Keep y as random for depth effect

        # Compute Perlin noise based on x and z
        noise_val = pnoise2(x * PERLIN_SCALE, z * PERLIN_SCALE, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=10240, repeaty=10240, base=0)

        # Normalize noise_val from [-1, 1] to [0, 1]
        noise_norm = (noise_val + 1) / 2

        # Determine if the particle should be part of a cluster based on noise value
        # Higher noise_norm -> higher probability to be in a cluster
        cluster_probability = 10  # Adjust to control clustering intensity
        if random.random() < noise_norm * cluster_probability:
            # Apply a small offset to cluster around the original position
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, CLUSTER_OFFSET)
            x += radius * math.cos(angle)
            z += radius * math.sin(angle)

        return [x, y, z]

    shapes = [
        Box(
            center=generate_perlin_position(),
            dimensions=[1, 2, 1],
            color=(255, 255, 255)
        ) 
        for _ in range(NUM_CUBES)
    ]

    # ---------------------
    # VIDEO WRITER
    # ---------------------
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

    # Initialize frame buffer
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Simulation loop
    frames = int(FPS * DURATION)
    start_time = time.time()

    for frame_idx in range(frames):
        # 1) Apply semi-transparent black overlay (motion blur)
        overlay = np.zeros_like(frame, dtype=np.uint8)
        alpha = 0.9
        cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0, frame)

        # 2) Create temporary buffer for new shapes
        temp_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # 3) Sort shapes by some approximate depth (optional)
        shapes.sort(key=lambda s: s.center[1], reverse=True)

        # 4) Get camera rotation for this frame
        camera_rotation = get_camera_rotation(frame_idx)

        # 5) Draw & update shapes
        for shape in shapes:
            draw_shape(frame, shape, temp_buffer, camera_rotation)

            # Update shape position (falling effect)
            shape.center[1] -= 3  # tweak fall speed
            if shape.center[1] < -HEIGHT // 2:
                # Reset y position using Perlin noise again to maintain structure
                # Alternatively, assign a new Perlin noise-based position
                new_x, new_z = random.uniform(-4000, 4000), random.uniform(-4000, 4000)
                y = random.uniform(2800, 4200)

                # Compute Perlin noise for the new position
                noise_val = pnoise2(new_x * PERLIN_SCALE, new_z * PERLIN_SCALE, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=10240, repeaty=10240, base=0)
                noise_norm = (noise_val + 1) / 2

                # Determine if the particle should be part of a cluster
                if random.random() < noise_norm * 0.7:
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.uniform(0, CLUSTER_OFFSET)
                    new_x += radius * math.cos(angle)
                    new_z += radius * math.sin(angle)

                shape.center = np.array([new_x, y, new_z], dtype=np.float32)

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
            time_remaining = 0
            if frame_idx > 0:
                time_remaining = (elapsed_time / frame_idx) * (frames - frame_idx)
            print(f"Frame {frame_idx}/{frames} ({progress:.2f}%): Elapsed={elapsed_time:.2f}s, ETA={time_remaining:.2f}s")

        # 8) (Optional) Display the framebuffer
        cv2.imshow('Framebuffer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {OUTPUT_FILE}")

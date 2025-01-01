import cv2
import numpy as np
import random
import time

for _ in range(0,10):

    # Constants
    WIDTH, HEIGHT = 1920, 1080  # Reduced resolution for efficiency
    FPS = 60                    # Reduced FPS for testing
    DURATION = 60*60*1          # Shortened duration for testing (seconds)
    NUM_CUBES = random.randint(500, 1000)  # Random number of cubes
    OUTPUT_FILE = f'starfield_screensaver_{round(time.time())}.mp4'

    # Perspective projection factor
    PERSPECTIVE = 600
    CENTER_FACTOR = np.array([WIDTH // 2, HEIGHT // 2])

    # Cube class
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
            ])

    # Perspective projection
    def project(vertices):
        factors = PERSPECTIVE / (vertices[:, 1] + PERSPECTIVE)
        projected = vertices[:, [0, 2]] * factors[:, None]
        return (projected + CENTER_FACTOR).astype(int)

    # Draw shape with anti-aliasing
    def draw_shape(frame, shape, temp_buffer):
        vertices = shape.vertices()
        projected_vertices = project(vertices)

        faces = [
            ([0, 1, 2, 3], np.mean(vertices[[0, 1, 2, 3], 2])),
            ([4, 5, 6, 7], np.mean(vertices[[4, 5, 6, 7], 2])),
            ([0, 1, 5, 4], np.mean(vertices[[0, 1, 5, 4], 2])),
            ([2, 3, 7, 6], np.mean(vertices[[2, 3, 7, 6], 2])),
            ([1, 2, 6, 5], np.mean(vertices[[1, 2, 6, 5], 2])),
            ([0, 3, 7, 4], np.mean(vertices[[0, 3, 7, 4], 2])),
        ]

        # Sort faces by depth for proper rendering
        faces.sort(key=lambda face: face[1], reverse=True)

        for face, _ in faces:
            pts = np.array([projected_vertices[i] for i in face], np.int32)
            pts = pts.reshape((-1, 1, 2))
            # Use LINE_AA for anti-aliasing if possible
            cv2.fillConvexPoly(temp_buffer, pts, shape.color, lineType=cv2.LINE_AA)

    # Generate shapes
    shapes = [
        Box(
            center=[
                random.uniform(-WIDTH // 2, WIDTH // 2),
                random.uniform(-200, 300),
                random.uniform(-WIDTH // 2, WIDTH // 2)
            ],
            dimensions=[1, 5, 1],
            color=(255, 255, 255)
        ) for _ in range(NUM_CUBES)
    ]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

    # Initialize frame buffer with black frame
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Simulation loop
    frames = int(FPS * DURATION)
    start_time = time.time()

    for frame_idx in range(frames):
        # Step 1: Apply semi-transparent black overlay for motion blur
        overlay = np.zeros_like(frame, dtype=np.uint8)
        alpha = 0.95  # 95% of the previous frame remains
        cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0, frame)

        # Step 2: Create temporary buffer for new shapes
        temp_buffer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Sort shapes by Z-depth
        shapes.sort(key=lambda s: s.center[2], reverse=True)

        # Draw and update shapes
        for shape in shapes:
            draw_shape(frame, shape, temp_buffer)
            # Update shape position (falling effect)
            shape.center[1] -= 4  # Fall speed
            if shape.center[1] < -HEIGHT // 2:
                shape.center[1] = random.uniform(1000, 1600)  # Reset position

        # Step 3: Apply glow effect
        # Blur the temp buffer to create glow
        blurred = cv2.GaussianBlur(temp_buffer, (21, 21), sigmaX=0, sigmaY=0)
        # Composite the blurred image with the temp buffer
        glow = cv2.addWeighted(temp_buffer, 1.0, blurred, 0.5, 0)
        # Add the glow to the main frame
        cv2.add(frame, glow, frame)

        # Write frame to video
        out.write(frame)

        # Display statistics every 60 frames
        if frame_idx % 60 == 0:
            elapsed_time = time.time() - start_time
            progress = (frame_idx / frames) * 100
            if frame_idx > 0:
                time_remaining = (elapsed_time / frame_idx) * (frames - frame_idx)
            else:
                time_remaining = 0
            print(
                f"Frame {frame_idx}/{frames} ({progress:.2f}% complete) - "
                f"Elapsed: {elapsed_time:.2f}s, Remaining: {time_remaining:.2f}s"
            )

        # Show framebuffer (optional)
        cv2.imshow('Framebuffer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    out.release()
    cv2.destroyAllWindows()
    print(f'Video saved to {OUTPUT_FILE}')


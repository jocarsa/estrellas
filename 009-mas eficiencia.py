import cv2
import numpy as np
import random
import time
from multiprocessing.pool import ThreadPool

# Constants
WIDTH, HEIGHT = 1920, 1080
FPS = 60
DURATION = 60*60  # seconds
NUM_CUBES = 1400
OUTPUT_FILE = 'falling_boxes_' + str(round(time.time())) + '.mp4'

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

# Draw shape
def draw_shape(frame, shape):
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

    faces.sort(key=lambda face: face[1], reverse=True)

    for face, _ in faces:
        pts = np.array([projected_vertices[i] for i in face], np.int32)
        cv2.fillConvexPoly(frame, pts, shape.color)

# Generate shapes
shapes = [
    Box(
        center=[
            random.uniform(-WIDTH // 2, WIDTH // 2),
            random.uniform(-200, 300),
            random.uniform(-WIDTH // 2, WIDTH // 2)
        ],
        dimensions=[1, 5, 1],
        color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    ) for _ in range(NUM_CUBES)
]

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

# Simulation loop
frames = int(FPS * DURATION)
start_time = time.time()

for frame_idx in range(frames):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Sort shapes by Z-depth
    shapes.sort(key=lambda s: s.center[2], reverse=True)

    # Draw and update shapes
    for shape in shapes:
        draw_shape(frame, shape)
        shape.center[1] -= 4  # Fall speed
        if shape.center[1] < -HEIGHT // 2:
            shape.center[1] = random.uniform(1000, 1600)  # Reset position

    # Write frame to video
    out.write(frame)

    # Display statistics every 60 frames
    if frame_idx % 60 == 0:
        elapsed_time = time.time() - start_time
        progress = (frame_idx / frames) * 100
        time_remaining = (elapsed_time / (frame_idx + 1)) * (frames - frame_idx - 1)
        print(
            f"Frame {frame_idx}/{frames} ({progress:.2f}% complete) - "
            f"Elapsed: {elapsed_time:.2f}s, Remaining: {time_remaining:.2f}s"
        )
    # Show framebuffer
        cv2.imshow('Framebuffer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
out.release()
cv2.destroyAllWindows()
print(f'Video saved to {OUTPUT_FILE}')

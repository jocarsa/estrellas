import cv2
import numpy as np
import random

# Constants
WIDTH, HEIGHT = 1920, 1080
FPS = 60
DURATION = 10  # seconds
NUM_CUBES = 400
OUTPUT_FILE = 'falling_cubes.mp4'

# Cube properties
class Cube:
    def __init__(self, center, size, color):
        self.center = np.array(center, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.color = color

    def vertices(self):
        x, y, z = self.center
        sx, sy, sz = self.size
        return [
            [x - sx, y - sy, z - sz],
            [x + sx, y - sy, z - sz],
            [x + sx, y + sy, z - sz],
            [x - sx, y + sy, z - sz],
            [x - sx, y - sy, z + sz],
            [x + sx, y - sy, z + sz],
            [x + sx, y + sy, z + sz],
            [x - sx, y + sy, z + sz],
        ]

# Perspective projection
def project(vertex):
    persp = 600
    x, y, z = vertex
    factor = persp / (y + persp)
    return int(WIDTH // 2 + x * factor), int(HEIGHT // 2 - z * factor)

# Draw a cube
def draw_cube(frame, cube):
    vertices = [project(v) for v in cube.vertices()]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Connecting edges
    ]
    for start, end in edges:
        cv2.line(frame, vertices[start], vertices[end], cube.color, 2)

# Create cubes
cubes = []
for _ in range(NUM_CUBES):
    x = random.uniform(-WIDTH // 2, WIDTH // 2)
    y = random.uniform(-200, 300)
    z = random.uniform(-WIDTH // 2, WIDTH // 2)
    size = random.uniform(5, 15)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cubes.append(Cube([x, y, z], [size, size, size], color))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

# Simulation loop
frames = int(FPS * DURATION)
for _ in range(frames):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for cube in cubes:
        draw_cube(frame, cube)
        # Update cube position
        cube.center[1] -= 4  # Fall speed
        if cube.center[1] < -HEIGHT // 2:
            cube.center[1] = random.uniform(300, 600)  # Reset position
    out.write(frame)

    # Show framebuffer
    cv2.imshow('Framebuffer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
out.release()
cv2.destroyAllWindows()
print(f'Video saved to {OUTPUT_FILE}')

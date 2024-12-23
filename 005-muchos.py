import cv2
import numpy as np
import random
import time

# Constants
WIDTH, HEIGHT = 1920, 1080
FPS = 60
DURATION = 60*60  # seconds
NUM_CUBES = 400
OUTPUT_FILE = 'falling_boxes'+str(round(time.time()))+'.mp4'

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

# Box class
class Box:
    def __init__(self, center, dimensions, color):
        self.center = np.array(center, dtype=np.float32)
        self.dimensions = np.array(dimensions, dtype=np.float32)
        self.color = color

    def vertices(self):
        x, y, z = self.center
        dx, dy, dz = self.dimensions
        return [
            [x - dx, y - dy, z - dz],
            [x + dx, y - dy, z - dz],
            [x + dx, y + dy, z - dz],
            [x - dx, y + dy, z - dz],
            [x - dx, y - dy, z + dz],
            [x + dx, y - dy, z + dz],
            [x + dx, y + dy, z + dz],
            [x - dx, y + dy, z + dz],
        ]

# Perspective projection
def project(vertex):
    persp = 600
    x, y, z = vertex
    factor = persp / (y + persp)
    return int(WIDTH // 2 + x * factor), int(HEIGHT // 2 - z * factor)

# Draw a shape (Cube or Box)
def draw_shape(frame, shape):
    vertices = [project(v) for v in shape.vertices()]
    faces = [
        ([0, 1, 2, 3], np.mean([shape.vertices()[i][2] for i in [0, 1, 2, 3]])),  # Bottom face
        ([4, 5, 6, 7], np.mean([shape.vertices()[i][2] for i in [4, 5, 6, 7]])),  # Top face
        ([0, 1, 5, 4], np.mean([shape.vertices()[i][2] for i in [0, 1, 5, 4]])),  # Front face
        ([2, 3, 7, 6], np.mean([shape.vertices()[i][2] for i in [2, 3, 7, 6]])),  # Back face
        ([1, 2, 6, 5], np.mean([shape.vertices()[i][2] for i in [1, 2, 6, 5]])),  # Right face
        ([0, 3, 7, 4], np.mean([shape.vertices()[i][2] for i in [0, 3, 7, 4]])),  # Left face
    ]
    faces.sort(key=lambda face: face[1], reverse=True)
    for face, _ in faces:
        pts = np.array([vertices[i] for i in face], np.int32)
        cv2.fillConvexPoly(frame, pts, shape.color)

# Create boxes
shapes = []
for _ in range(NUM_CUBES):
    x = random.uniform(-WIDTH // 2, WIDTH // 2)
    y = random.uniform(-200, 300)
    z = random.uniform(-WIDTH // 2, WIDTH // 2)
    dimensions = [1, 5, 1]
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    shapes.append(Box([x, y, z], dimensions, color))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

# Simulation loop
frames = int(FPS * DURATION)
for _ in range(frames):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Sort shapes by Z-depth (farthest to nearest)
    shapes.sort(key=lambda s: s.center[2], reverse=True)

    for shape in shapes:
        draw_shape(frame, shape)
        # Update shape position
        shape.center[1] -= 4  # Fall speed
        if shape.center[1] < -HEIGHT // 2:
            shape.center[1] = random.uniform(1000, 1600)  # Reset position
    out.write(frame)

    # Show framebuffer
    cv2.imshow('Framebuffer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
out.release()
cv2.destroyAllWindows()
print(f'Video saved to {OUTPUT_FILE}')

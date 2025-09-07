import cv2
import matplotlib.pyplot as plt
import numpy as np

from seamless import grow_image


def load(path):
    """Load a color image and normalize values to [0, 1] range"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img

def save(img, path):
    """Save a color image"""
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def initialize_image_from_seed(sample, size=(256, 256), seed_size=3):
    h, w, c = sample.shape
    sy, sx = np.random.randint(0, h - seed_size), np.random.randint(0, w - seed_size)
    seed_patch = sample[sy:sy+seed_size, sx:sx+seed_size, :]

    image = -np.ones((*size, c))
    center = (size[1]//2, size[0]//2)
    start_x = center[0] - seed_size // 2
    start_y = center[1] - seed_size // 2

    image[start_y:start_y+seed_size, start_x:start_x+seed_size, :] = seed_patch
    return image

# Load sample image
sample = load("img.png")

# Initialize target image with a random seed from the sample
image = initialize_image_from_seed(sample, size=(512, 512))

# Run the algorithm
window_size = 7
result = grow_image(sample, image, window_size)

# Save result
save(result, "output_synthesized_image_color.jpg")

# Display
plt.imshow(result)
plt.title("Synthesized Color Image")
plt.axis("off")
plt.show()

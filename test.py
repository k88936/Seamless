import cv2
import matplotlib.pyplot as plt
import numpy as np

from seamless import grow_image


def load_grayscale_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return img

def save_grayscale_image(img, path):
    cv2.imwrite(path, (img * 255).astype(np.uint8))

def initialize_image_from_seed(sample, size=(256, 256), seed_size=3):
    h, w = sample.shape
    sy, sx = np.random.randint(0, h - seed_size), np.random.randint(0, w - seed_size)
    seed_patch = sample[sy:sy+seed_size, sx:sx+seed_size]

    image = -np.ones(size)
    center = (size[1]//2, size[0]//2)
    start_x = center[0] - seed_size // 2
    start_y = center[1] - seed_size // 2

    image[start_y:start_y+seed_size, start_x:start_x+seed_size] = seed_patch
    return image

# Load sample image
sample = load_grayscale_image("img.png")

# Initialize target image with a random seed from the sample
image = initialize_image_from_seed(sample, size=(128, 128))

# Run the algorithm
window_size = 7
result = grow_image(sample, image, window_size)

# Save result
save_grayscale_image(result, "output_synthesized_image.jpg")

# Display
plt.imshow(result, cmap='gray')
plt.title("Synthesized Image")
plt.axis("off")
plt.show()
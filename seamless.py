import numpy as np
import cv2
import psutil
import os
from scipy.ndimage import binary_dilation
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Constants
ERR_THRESHOLD = 0.1
MAX_ERR_THRESHOLD = 0.3
SIGMA_RATIO = 6.4  # Sigma = WindowSize / SIGMA_RATIO


def get_gaussian_mask(window_size):
    """Generate a normalized 2D Gaussian mask."""
    sigma = window_size / SIGMA_RATIO
    kernel = cv2.getGaussianKernel(window_size, sigma)
    return np.outer(kernel, kernel)


def get_neighborhood_window(image, pixel, window_size):
    """Return a window around the given pixel."""
    h, w, c = image.shape
    x, y = pixel
    pad = window_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=-1)
    return padded_image[y:y + window_size, x:x + window_size, :]


def find_matches(template, sample_image, valid_mask, gauss_mask, tot_weight):
    """
    Find all pixels in sample_image whose neighborhoods match the template within threshold.
    """
    window_size = template.shape[0]
    h, w, c = sample_image.shape
    ssd_map = np.zeros_like(sample_image[..., 0], dtype=np.float32)

    pad = window_size // 2
    padded_sample = np.pad(sample_image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    def compute_ssd(args):
        """Compute SSD for a single patch."""
        i, j = args
        patch = padded_sample[i:i + window_size, j:j + window_size, :]
        diff = (template - patch) * valid_mask[..., np.newaxis]
        weighted_ssd = np.sum((diff ** 2) * gauss_mask[..., np.newaxis] * valid_mask[..., np.newaxis])
        if tot_weight > 0:
            return i, j, weighted_ssd / tot_weight
        else:
            return i, j, float('inf')

    for i, j  in product(range(h), range(w)):
        _,_,ssd = compute_ssd((i, j))
        ssd_map[i, j] = ssd



    min_ssd = np.min(ssd_map[np.isfinite(ssd_map)])
    threshold = min_ssd * (1 + ERR_THRESHOLD)
    candidates = np.where(ssd_map <= threshold)
    indices = list(zip(candidates[0], candidates[1]))

    return [sample_image[i, j, :] for i, j in indices]


def get_unfilled_neighbors(image):
    """
    Return list of unfilled pixels that have at least one filled neighbor,
    sorted by number of neighbors descending, and then randomly permuted.
    """
    filled = np.any(image != -1, axis=2).astype(np.uint8)
    unfilled = (filled == 0).astype(np.uint8)
    dilated = binary_dilation(filled, structure=np.ones((3, 3)))
    edge = dilated & unfilled
    coords = np.argwhere(edge)

    # Sort by number of filled neighbors
    def filled_neighbor_count(p):
        x, y = p
        neighbors = filled[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        return np.sum(neighbors)

    coords = sorted(coords, key=filled_neighbor_count, reverse=True)
    np.random.shuffle(coords)  # Randomize order among same-count groups
    return [tuple(p) for p in coords]


def grow_image(sample_image, image, window_size):
    """
    Grow the image based on template matching from the sample image.
    """
    import time
    start_time = time.time()
    height, width, channels = image.shape
    gauss_mask = get_gaussian_mask(window_size)
    current_threshold = MAX_ERR_THRESHOLD
    max_iterations = 1000  # Prevent infinite loops
    iteration = 0
    total_pixels = np.sum(np.sum(image == -1, axis=2) == 3)  # Count unfilled pixels

    # Initialize resource monitoring
    process = psutil.Process(os.getpid())

    while True:
        iteration += 1
        if iteration > max_iterations:
            print("Reached max iterations.")
            break

        pixel_list = get_unfilled_neighbors(image)
        if not pixel_list:
            elapsed_time = time.time() - start_time
            print(f"Filled completely in {elapsed_time:.2f} seconds.")
            break

        # Calculate progress percentage
        remaining_pixels = np.sum(np.sum(image == -1, axis=2) == 3)
        filled_pixels = total_pixels - remaining_pixels
        progress_percent = (filled_pixels / total_pixels) * 100
        elapsed_time = time.time() - start_time
        print(
            f"Processing: {progress_percent:.2f}% complete, {remaining_pixels} pixels remaining, time: {elapsed_time:.2f}s")

        progress = False
        for x, y in pixel_list:
            template = get_neighborhood_window(image, (x, y), window_size)
            valid_mask = np.any(template != -1, axis=2).astype(float)
            tot_weight = np.sum(gauss_mask * valid_mask)

            matches = find_matches(template, sample_image, valid_mask, gauss_mask, tot_weight)
            if not matches:
                continue

            matched_value = matches[np.random.choice(len(matches))]
            image[y, x, :] = matched_value
            progress = True

        if not progress:
            current_threshold *= 1.1
            print(f"Increasing threshold: {current_threshold:.3f}")
        else:
            print(f"Progress made. Threshold: {current_threshold:.3f}")

        total_time = time.time() - start_time
        pixels_per_second = filled_pixels / total_time if total_time > 0 else 0
        print(f"Processing speed: {pixels_per_second:.2f} pixels/second")
    return image

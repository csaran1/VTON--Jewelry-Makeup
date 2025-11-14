import cv2
import numpy as np

def segment_jewellery(image_path):
    """
    Segments jewellery from an image using K-Means clustering.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - original (numpy array): Original image in RGB format.
    - mask (numpy array): Binary mask where jewellery is separated.
    - segmented (numpy array): Jewellery image with transparent background.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    print("image_shape",image.shape)
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape image for clustering
    pixels = image_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Apply K-Means clustering
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert labels into a mask
    labels = labels.reshape(image_rgb.shape[:2])
    unique, counts = np.unique(labels, return_counts=True)
    bg_label = unique[np.argmax(counts)]

    # Create binary mask
    mask = np.where(labels == bg_label, 0, 255).astype(np.uint8)

    # Apply mask to the original image
    b, g, r = cv2.split(image_rgb)
    alpha = mask  # Use stored mask as alpha channel
    segmented = cv2.merge((b, g, r, alpha))

    print("segment_file_mask", mask.shape[0], mask.shape[1])
    print("segment file_image_rgb", image_rgb.shape[0], image_rgb.shape[1])
    print("segment file_segmented", segmented.shape[0], segmented.shape[1])

    return image_rgb, mask, segmented


image_path = "image1.webp"
original, mask, segmented = segment_jewellery(image_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Generated Mask")
axes[1].axis("off")

axes[2].imshow(segmented)
axes[2].set_title("Segmented Image")
axes[2].axis("off")

plt.show()
cv2.imwrite("jewellery_segment1.png", cv2.cvtColor(segmented, cv2.COLOR_RGBA2BGRA))
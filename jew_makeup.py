import sys
import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import mediapipe as mp

# Add EleGANt to the system path
elegant_path = r'D:\try on\EleGANt'
sys.path.append(elegant_path)

# Import EleGANt modules
from EleGANt.training.config import get_config
from EleGANt.training.inference import Inference

# MediaPipe setup for face landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)


# Initialize EleGANt configuration
class Args:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = r'D:\try on\EleGANt\ckpts\sow_pyramid_a5_e3d2_remapped.pth'
    save_folder = ''  # Adjust path if needed


args = Args()
config = get_config()


def transfer_makeup(source_image, reference_image):
    # Initialize the Inference object
    inference = Inference(config=config, args=args, model_path=args.model_path)

    src_img_np = np.array(source_image)
    ref_img_np = np.array(reference_image)

    print(f"Original source image size: {src_img_np.shape}")
    print(f"Original reference image size: {ref_img_np.shape}")

    # Resize reference image to match source image dimensions
    ref_img_resized_np = cv2.resize(ref_img_np, (src_img_np.shape[1], src_img_np.shape[0]))
    print(f"Resized reference image size: {ref_img_resized_np.shape}")

    src_img_pil = Image.fromarray(src_img_np)
    ref_img_resized_pil = Image.fromarray(ref_img_resized_np)

    if hasattr(inference, 'transfer'):
        output_img_pil = inference.transfer(src_img_pil, ref_img_resized_pil)
        print(f"Output image size after transfer: {output_img_pil.size}")

        output_img_np = np.array(output_img_pil)
        print(f"Output image numpy array size: {output_img_np.shape}")

        if output_img_np.shape != src_img_np.shape:
            print("Output image size does not match source image size. Resizing output image.")
            # Resize the output image to match the source image dimensions
            output_img_np_resized = cv2.resize(output_img_np, (src_img_np.shape[1], src_img_np.shape[0]))
            return output_img_np_resized
        else:
            return output_img_np
    else:
        print("Transfer method not found in Inference module.")
        return None


def segment_jewellery(image):
    """
    Segments jewellery from an image using K-Means clustering.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - original (numpy array): Original image in RGB format.
    - mask (numpy array): Binary mask where jewellery is separated.
    - segmented (numpy array): Jewellery image with transparent background.
    """
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    print("\n\nimage shape",image.shape)
    h, w, _ = image.shape
    print("image height and width ", h, w)
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape image for clustering
    pixels = image_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)

    # K-Means clustering
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Converting labels into a mask
    labels = labels.reshape(image_rgb.shape[:2])
    unique, counts = np.unique(labels, return_counts=True)
    bg_label = unique[np.argmax(counts)]

    # Creating binary mask
    mask = np.where(labels == bg_label, 0, 255).astype(np.uint8)

    # Applying mask to the original image
    b, g, r = cv2.split(image_rgb)
    alpha = mask  # Use stored mask as alpha channel
    segmented = cv2.merge((b, g, r, alpha))
    segmented = cv2.cvtColor(segmented, cv2.COLOR_RGBA2BGRA)
    #cv2.imshow(segmented)
    return image_rgb, mask, segmented


# image_path = "image1.webp"
# original, mask, segmented = segment_jewellery(image_path)


import numpy as np
import cv2
from mediapipe.python.solutions.face_mesh import FaceMesh

# Global variables for dragging
dragging = False
necklace_position = (0, 0)
necklace_resized = None
scale_factor = 1.0  # Scale factor for resizing image


def on_mouse(event, x, y, flags, param):
    global dragging, necklace_position

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True  # Start dragging
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        necklace_position = (int(x / scale_factor), int(y / scale_factor))
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False  # Stop dragging


def detect_neck_and_adjust_necklaces(image, necklace_file):
    global necklace_position, necklace_resized, scale_factor

    # Resize image for better viewing
    max_display_size = 500  # Max width or height
    h, w, _ = image.shape
    if max(h, w) > max_display_size:
        scale_factor = max_display_size / max(h, w)
        image_resized = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    else:
        image_resized = image.copy()

    # Read necklace image
    necklace_bytes = np.frombuffer(necklace_file.read(), np.uint8)
    necklace = cv2.imdecode(necklace_bytes, cv2.IMREAD_UNCHANGED)
    original, mask, necklace = segment_jewellery(necklace)
    if necklace is None:
        raise ValueError("Failed to load necklace image. Ensure it is a valid transparent PNG format.")

    # Initialize MediaPipe Face Mesh
    face_mesh = FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        #print('landmark', landmarks)
        '''jaw_indices = [152, 377, 378, 379, 365, 367, 164, 0, 17, 18, 200, 199, 175, 152]
        for i, landmark in enumerate(landmarks):
            if i in jaw_indices:
                for landmark in landmarks:

                # Convert the landmark's normalized coordinates to pixel values
                    h, w, c = rgb_image.shape  # Height, width, and channels of the image
                    x, y = int(landmark.x * w), int(landmark.y * h)

               # Draw a circle at the landmark's location
                    cv2.circle(rgb_image, (x, y), 2, (0, 255, 0), -1)  # Green
        cv2.imshow("Image with Landmarks", rgb_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        # Get chin and jaw positions
        chin_index = 152
        left_jaw_index = 234
        right_jaw_index = 454

        #chin_x = int(landmarks[chin_index].x * w)
        #chin_y = int(landmarks[chin_index].y * h)
        #left_jaw_x = int(landmarks[left_jaw_index].x * w)
        #right_jaw_x = int(landmarks[right_jaw_index].x * w)

        chin_x = int(landmarks[chin_index].x * w)
        chin_y = int(landmarks[chin_index].y * h)
        left_jaw_x = int(landmarks[left_jaw_index].x * w)
        left_jaw_y = int(landmarks[left_jaw_index].y * h)
        right_jaw_x = int(landmarks[right_jaw_index].x * w)
        right_jaw_y = int(landmarks[right_jaw_index].y * h)

        neck_width = abs(right_jaw_x - left_jaw_x)
        print("Neck Width: ",neck_width)
        y_offset = chin_y + int(0.02 * h)  # Slightly below chin
        print("Y_offset:",y_offset)
        '''pts=True
        if(pts==True):
            # Draw circles on the jawline points
            cv2.circle(rgb_image, (left_jaw_x, left_jaw_y), 5, (255, 0, 0), -1)  # Blue dot for left jaw
            cv2.circle(rgb_image, (right_jaw_x, right_jaw_y), 5, (255, 0, 0), -1)  # Blue dot for right jaw

            # Draw circles where the necklace should be placed
            cv2.circle(rgb_image, (left_jaw_x, y_offset), 5, (255, 0, 0), -1)  # Blue dot for left side
            cv2.circle(rgb_image, (right_jaw_x, y_offset), 5, (255, 0, 0), -1)  # Blue dot for right side

            # Draw line where the necklace should be placed
            cv2.line(image, (left_jaw_x, y_offset), (right_jaw_x, y_offset), (0, 255, 0),
                     2)  # Green line for correct neck width

            # Show the image with the landmarks
            cv2.imshow("Neck Width Visualization", rgb_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''

        # Resize necklace to match neck width
        necklace_resized = cv2.resize(necklace, (neck_width, int(neck_width * necklace.shape[0] / necklace.shape[1])))

        # Set initial position
        necklace_position = (chin_x - necklace_resized.shape[1] // 2, y_offset)
    else:
        raise ValueError("No face detected in the input image.")

    # Create window and set mouse callback
    cv2.namedWindow("Adjust Necklace", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Adjust Necklace", on_mouse)

    while True:
        overlay = image_resized.copy()

        # Place necklace at updated position (resized for display)
        x_display, y_display = int(necklace_position[0] * scale_factor), int(necklace_position[1] * scale_factor)
        necklace_display = cv2.resize(necklace_resized, (int(necklace_resized.shape[1] * scale_factor),
                                                         int(necklace_resized.shape[0] * scale_factor)))

        overlay_image_with_masks(overlay, necklace_display, x_offset=x_display, y_offset=y_display)

        cv2.imshow("Adjust Necklace", overlay)
        key = cv2.waitKey(1)

        # Press 's' to save final result
        if key == ord('s'):
            # Map back to original size before applying the necklace
            final_overlay = image.copy()
            overlay_image_with_masks(final_overlay, necklace_resized,
                                     x_offset=necklace_position[0], y_offset=necklace_position[1])
            cv2.destroyAllWindows()
            return final_overlay
        # Press 'q' to exit without saving
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return image


def overlay_image_with_masks(background, overlay, x_offset, y_offset):
    """ Overlays an image with transparency handling """
    h, w, _ = background.shape
    h_o, w_o, c_o = overlay.shape

    if c_o < 4:
        return  # To Ensure overlay has an alpha channel

    # Ensure valid positioning
    if x_offset < 0 or y_offset < 0 or x_offset + w_o > w or y_offset + h_o > h:
        return

    # Extract overlay channels
    overlay_rgb = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0  # Normalize alpha channel

    # Blend images
    for c in range(3):  # Blend R, G, B channels
        background[y_offset:y_offset + h_o, x_offset:x_offset + w_o, c] = (
                (1 - mask) * background[y_offset:y_offset + h_o, x_offset:x_offset + w_o, c] +
                mask * overlay_rgb[:, :, c]
        )


def find_necklace_3point(mask):
    print("mask shape", mask.shape[0], mask.shape[1])

    first_non_zero = None
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] != 0:
                first_non_zero = (x, y)
                break
        if first_non_zero:
            break

    print("First non-zero pixel (x, y):", first_non_zero)

    # Find second non-zero pixel (right to left, top to bottom)
    second_non_zero = None
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1] - 1, 0, -1):
            if mask[y, x] != 0:
                second_non_zero = (x, y)
                break
        if second_non_zero:
            break

    print("Second non-zero pixel (x, y):", second_non_zero)

    third_non_zero = None
    for y in range(mask.shape[0]-1, 0, -1):
        for x in range(mask.shape[1]):
            if mask[y, x] != 0:
                third_non_zero = (x, y)
                break
        if third_non_zero:
            break

    print("third non-zero pixel coordinates (x, y):", third_non_zero)

    vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Draw endpoints and line
    #cv2.circle(vis_mask, first_non_zero, 5, (0, 0, 255), -1)  # Red
    #cv2.circle(vis_mask, second_non_zero, 5, (0, 255, 0), -1)  # Green
    #cv2.line(vis_mask, first_non_zero, second_non_zero, (255, 0, 0), 2)

    #cv2.imwrite("vis_mask.jpg",vis_mask)

    return first_non_zero,second_non_zero,third_non_zero

def detect_neck_and_adjust_necklace(image, necklace_file):
    import numpy as np
    import cv2
    from mediapipe.python.solutions.face_mesh import FaceMesh

    # # Read necklace image as a transparent PNG
    necklace_bytes = np.frombuffer(necklace_file.read(), np.uint8)
    necklace = cv2.imdecode(necklace_bytes, cv2.IMREAD_UNCHANGED)
    #print(necklace)
    original, mask, necklace = segment_jewellery(necklace)
    cv2.imshow("mask",mask)
    #necklace_width=find_necklace_width(mask)
    #print("necklace width",necklace_width)
    print("mask shape", mask.shape[0], mask.shape[1])
    necklace_point1,necklace_point2,necklace_point3 = find_necklace_3point(mask)
    print("Necklace_point1",necklace_point1)
    print("Necklace_point2",necklace_point2)
    print("Necklace_point3", necklace_point3)
    necklace_width=(necklace_point2[0]-necklace_point1[0])
    print("necklace width", necklace_width)

    print(necklace_point1[1])#height
    print(necklace_point2[1])
    print(necklace_point1[0])
    print(necklace_point3[0])#width


    necklace= necklace[necklace_point1[1]:necklace_point3[1],]
    #print(crop)
    #cv2.imshow("cropped",crop)
    #cv2.imwrite("cropped2.jpg", crop)
    #cv2.waitkey(0)

    if necklace is None:
        raise ValueError("Failed to load necklace image. Ensure it is a valid transparent PNG format.")

    # Initialize MediaPipe Face Mesh
    face_mesh = FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    h, w, _ = image.shape
    #image coordinates here are the boundang box of the necklace
    print("image height",h)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Neck-related landmark indices (adjusted for better placement)
        chin_index = 152  # Chin tip index-should be fixed
        left_jaw_index = 234  # Left jaw index-should be fixed
        right_jaw_index = 454  # Right jaw index-should be fixed

        # Calculate neck landmark coordinates
        chin_x = int(landmarks[chin_index].x * w )
        chin_y = int(landmarks[chin_index].y * h)
        left_jaw_x = int(landmarks[left_jaw_index].x * w)
        right_jaw_x = int(landmarks[right_jaw_index].x * w)
        right_jaw_y = int(landmarks[right_jaw_index].y * h)
        left_jaw_y = int(landmarks[left_jaw_index].y * h)

        print("right_jaw", right_jaw_x, right_jaw_y)
        print("left_jaw", left_jaw_x, left_jaw_y)
        print("chin", chin_x, chin_y)

        #offset=abs(chin_y-left_jaw_y)*0.5
        #print('offset',offset)

        # Calculate neck width and position
        neck_width = abs(right_jaw_x - left_jaw_x)
        y_offset = chin_y + int(0.25 * neck_width) # Offset slightly below the chin for alignment
        print("Y_offset",y_offset)
        # fine neck hyperparameters of offset-- model with frontal face like model2 and 3
        #y_offset = chin_y + offset
#necklace_resized = cv2.resize(necklace, (neck_width+15, int(neck_width * necklace.shape[0] / necklace.shape[1])))int(necklace.shape[1]*neck_width/necklace_width)
        ratio=(neck_width/necklace_width)*0.95#fine neck hyperparameters
        print(ratio)

        necklace_resized = cv2.resize(necklace, (int(necklace.shape[1]*ratio), int(necklace.shape[0]*ratio)))
        #necklace_resized = cv2.resize(necklace, (neck_width+15, int(neck_width * necklace.shape[0] / necklace.shape[1])))
        #print("new",int(necklace.shape[0]*ratio),int(necklace.shape[1]*ratio))
        #print("previous",neck_width+15, int(neck_width * necklace.shape[0] / necklace.shape[1]))
        print("necklace",necklace.shape[0],necklace.shape[1])

        # Calculate x and y offsets for placement
        print("necklace width",necklace_resized.shape[1])
        print("necklace height", necklace_resized.shape[0])
        x_offset = chin_x - necklace_resized.shape[1] // 2
        print("necklace x_offset",x_offset)
        #y_offset = max(y_offset, h - necklace_resized.shape[0])#min()
        print("necklace Y_offset",y_offset)
        print("neck_width:",neck_width)
        print("necklace_positions calculated",x_offset,y_offset)
        # Draw key points on the image for visualization
        ##cv2.circle(image, (chin_x, chin_y), 3, (255, 0, 0), -1)  # Red dot for chin
        #cv2.circle(image, (left_jaw_x, chin_y), 3, (255, 0, 0), -1)  # Red dot for left jaw
        #cv2.circle(image, (right_jaw_x, chin_y), 3, (255, 0, 0), -1)  # Red dot for right jaw

        ##cv2.circle(image, (left_jaw_x, left_jaw_y), 5, (255, 0, 0), -1)
        ##cv2.circle(image, (right_jaw_x, right_jaw_y), 5, (255, 0, 0), -1)
        #cv2.circle(image, (x_offset, y_offset), 5, (255, 0, 0), -1)
        ##cv2.circle(image, (right_jaw_x, y_offset), 5, (255, 0, 0), -1)
        ##cv2.circle(image, (left_jaw_x, y_offset), 5, (255, 0, 0), -1)

        # Draw line where the necklace should be placed
        ##cv2.line(image, (left_jaw_x, y_offset), (right_jaw_x, y_offset), (0, 255, 0),1)  # Green line for correct neck width

        # Overlay necklace with transparency handling
        overlay_image_with_mask(image, necklace_resized, x_offset=x_offset, y_offset=y_offset)
        return image
    else:
        raise ValueError("No face detected in the input image.")


def overlay_image_with_mask(background, overlay, x_offset, y_offset):
    h, w, c = overlay.shape
    for i in range(h):
        for j in range(w):
            if x_offset + j >= background.shape[1] or y_offset + i >= background.shape[0]:
                continue
            alpha = overlay[i, j, 3] / 255.0  # Use the alpha channel for transparency
            if alpha > 0:
                background[y_offset + i, x_offset + j] = (
                        (1 - alpha) * background[y_offset + i, x_offset + j] + alpha * overlay[i, j, :3]
                )


def create_mask(points, landmarks, image_width, image_height, image):
    points = [(int(landmarks[p].x * image_width), int(landmarks[p].y * image_height)) for p in points]
    points = np.array(points, dtype=np.int32)
    mask = np.zeros(image.shape, dtype=np.uint8)
    return mask, points


concealer_n = 1


def apply_lipliner(image, landmarks, image_width, image_height, color):
    """
    Apply lipliner effect to an image based on facial landmarks.

    Parameters:
    - image: The input image (BGR format).
    - landmarks: Detected facial landmarks.
    - image_width: Width of the image.
    - image_height: Height of the image.
    - color: Tuple of BGR values for the lipliner color (e.g., (0, 0, 255) for red).

    Returns:
    - The image with the lipliner applied.
    """
    # Define the indices for the lips region in the facial landmarks
    lips_points = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61
    ]

    # Convert landmark coordinates to pixel values
    lips_points = np.array([
        (int(landmarks[point].x * image_width), int(landmarks[point].y * image_height))
        for point in lips_points
    ], np.int32)

    # Create a blank mask for the lipliner
    lipliner_mask = np.zeros(image.shape, dtype=np.uint8)

    # Draw the lipliner on the mask
    cv2.polylines(lipliner_mask, [lips_points], isClosed=True, color=color, thickness=2)

    # Blur the lipliner to make it more natural
    blurred_mask = cv2.GaussianBlur(lipliner_mask, (5, 5), 2)

    # Combine the original image with the blurred lipliner mask
    result = cv2.addWeighted(image, 1, blurred_mask, 0.5, 0)

    return result


# Function to convert hex color to RGB format
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return rgb_color


# Function to apply lipstick with improved blending
def apply_lipstick(image, landmarks, image_width, image_height, color):
    lips_points = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78
    ]
    mask, lips_points = create_mask(lips_points, landmarks, image_width, image_height, image)
    cv2.fillPoly(mask, [lips_points], color)
    mask = cv2.GaussianBlur(mask, (7, 7), 4)
    image = cv2.addWeighted(image, 1, mask, 0.3, 0)
    return image


def apply_undereye(image, landmarks, image_width, image_height, color):
    """
    Apply an under-eye effect to an image using facial landmarks.

    Parameters:
    - image: Input BGR image.
    - landmarks: List of facial landmarks detected by MediaPipe or similar.
    - image_width: Width of the input image.
    - image_height: Height of the input image.
    - color: Tuple (B, G, R) representing the color of the under-eye effect.

    Returns:
    - Image with under-eye effect applied.
    """
    # Indices for the bottom part of the eyes
    left_eye_bottom = [133, 155, 154, 153, 145, 144, 163, 7, 33]
    right_eye_bottom = [362, 382, 381, 380, 374, 373, 390, 249, 263]

    # Convert landmark coordinates to pixel values
    left_eye_bottom_points = np.array([
        (int(landmarks[point].x * image_width), int(landmarks[point].y * image_height))
        for point in left_eye_bottom
    ], np.int32)
    right_eye_bottom_points = np.array([
        (int(landmarks[point].x * image_width), int(landmarks[point].y * image_height))
        for point in right_eye_bottom
    ], np.int32)

    # Create a mask for under-eye regions
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Fill the under-eye regions on the mask
    cv2.fillPoly(mask, [left_eye_bottom_points], color=color)
    cv2.fillPoly(mask, [right_eye_bottom_points], color=color)

    # Apply Gaussian blur for a smooth effect
    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 10)

    # Blend the blurred mask with the original image
    output_image = cv2.addWeighted(image, 1, blurred_mask, 0.4, 0)

    return output_image


# Function to apply blush with improved blending
def apply_blush(image, landmarks, image_width, image_height, color):
    blush1_points = [
        280, 411, 371, 352, 345
    ]
    mask, blush1_points = create_mask(blush1_points, landmarks, image_width, image_height, image)
    cv2.fillPoly(mask, [blush1_points], color)
    mask = cv2.GaussianBlur(mask, (35, 35), 30)
    image = cv2.addWeighted(image, 1, mask, 0.15, 0)
    blush2_points = [
        187, 147, 137, 116, 50
    ]
    mask, blush2_points = create_mask(blush2_points, landmarks, image_width, image_height, image)
    cv2.fillPoly(mask, [blush2_points], color)
    mask = cv2.GaussianBlur(mask, (35, 35), 30)
    image = cv2.addWeighted(image, 1, mask, 0.15, 0)
    return image


# Function to apply concealer
def apply_concealer(image, landmarks, image_width, image_height, color, concealer_n):
    concealer_points = [
        83, 18, 313, 421, 428, 396, 175, 171, 208, 201
    ]
    if concealer_n == 2:
        concealer_points = [
            1, 45, 51, 3, 196, 122, 193, 108, 151, 337, 417, 351, 419, 248, 281, 275
        ]
    if concealer_n == 3:
        concealer_points = [
            412, 277, 266, 280, 345, 454, 356, 249, 390, 373, 374, 380, 381, 382, 362
        ]
        mask, concealer_points = create_mask(concealer_points, landmarks, image_width, image_height, image)
        cv2.fillPoly(mask, [concealer_points], color)
        mask = cv2.GaussianBlur(mask, (35, 35), 35)
        image = cv2.addWeighted(image, 1, mask, 0.15, 0)
        concealer_points = [133, 155, 154, 153, 145, 144, 163, 7, 33, 34, 227, 116, 50, 36, 47, 188]
    mask, concealer_points = create_mask(concealer_points, landmarks, image_width, image_height, image)
    cv2.fillPoly(mask, [concealer_points], color)
    mask = cv2.GaussianBlur(mask, (35, 35), 40)
    image = cv2.addWeighted(image, 1, mask, 0.15, 0)
    return image


# function to apply eyeshadow
def apply_eyeshadow(image, landmarks, image_width, image_height, color):
    eyeshadow_points = [
        414, 286, 258, 257, 259, 467, 445, 444, 443, 442, 441
    ]
    mask, eyeshadow_points = create_mask(eyeshadow_points, landmarks, image_width, image_height, image)
    cv2.fillPoly(mask, [eyeshadow_points], color)
    mask = cv2.GaussianBlur(mask, (25, 25), 15)
    image = cv2.addWeighted(image, 1, mask, 0.6, 0)
    eyeshadow_points = [
        190, 56, 28, 27, 29, 30, 225, 224, 223, 222, 221
    ]
    mask, eyeshadow_points = create_mask(eyeshadow_points, landmarks, image_width, image_height, image)
    cv2.fillPoly(mask, [eyeshadow_points], color)
    mask = cv2.GaussianBlur(mask, (25, 25), 15)
    image = cv2.addWeighted(image, 1, mask, 0.6, 0)
    return image


# Function to apply eyebrows with improved blending
def apply_eyebrows(image, landmarks, image_width, image_height, color):
    eyebrow_points = [
        55, 107, 66, 105, 63, 70, 53, 52, 55
    ]
    mask, eyebrow_points = create_mask(eyebrow_points, landmarks, image_width, image_height, image)
    cv2.fillPoly(mask, [eyebrow_points], color)
    mask = cv2.GaussianBlur(mask, (25, 25), 15)
    image = cv2.addWeighted(image, 1, mask, 0.2, 0)
    eyebrow_points = [
        285, 295, 282, 283, 276, 293, 334, 296, 336
    ]
    mask, eyebrow_points = create_mask(eyebrow_points, landmarks, image_width, image_height, image)
    cv2.fillPoly(mask, [eyebrow_points], color)
    mask = cv2.GaussianBlur(mask, (25, 25), 15)
    image = cv2.addWeighted(image, 1, mask, 0.2, 0)
    return image


# Function to apply skin toner with improved blending
def apply_skin_toner(image, landmarks, image_width, image_height, color):
    skin_points = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                   397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                   172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    mask, skin_points = create_mask(skin_points, landmarks, image_width, image_height, image)
    cv2.fillPoly(mask, [skin_points], color)
    mask = cv2.GaussianBlur(mask, (11, 11), 10)
    return cv2.addWeighted(image, 1, mask, 0.1, 0)


def apply_fine_tune_makeup(image, results, lipstick, lipstick_color, undereye, eye_liner_color, concealer,
                           concealer_color, concealer_n, blush, blush_color, toner, toner_color, eyebrows, lipliner,
                           lipliner_color, eyebrow_color, eyeshadow, eyeshadow_color, saturation, saturation_level):
    # image = cv2.resize(image,(800, 500))
    image = np.array(image)
    orig = image
    image_height, image_width, _ = image.shape
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            if toner:
                image = apply_skin_toner(image, landmarks, image_width, image_height, toner_color)
            if concealer:
                image = apply_concealer(image, landmarks, image_width, image_height, concealer_color,
                                        concealer_n)
            if undereye:
                image = apply_undereye(image, landmarks, image_width, image_height, eye_liner_color)
            if blush:
                image = apply_blush(image, landmarks, image_width, image_height, blush_color)
            if lipliner:
                image = apply_lipliner(image, landmarks, image_width, image_height, lipliner_color)
            if eyebrows:
                image = apply_eyebrows(image, landmarks, image_width, image_height, eyebrow_color)
            if eyeshadow:
                image = apply_eyeshadow(image, landmarks, image_width, image_height, eyeshadow_color)
            if lipstick:
                image = apply_lipstick(image, landmarks, image_width, image_height, lipstick_color)
            if saturation:
                image = adjust_saturation(image, saturation_level)

    return image


def adjust_saturation(image, saturation_level):
    """Adjusts the saturation of an image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Scale saturation: 50 is neutral, <50 reduces, >50 increases
    saturation_factor = saturation_level / 50.0
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)

    adjusted_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)


# if saturation and st.session_state.jewelry_output_image is not None:
#     st.session_state.jewelry_output_image = adjust_saturation(st.session_state.jewelry_output_image, saturation_level)
#     st.image(st.session_state.jewelry_output_image, caption="Saturation Adjusted Image", width=300)


def main():
    st.title("Integrated Makeup Transfer and Jewelry Try-On")
    st.header("Jewelry Try-On")

    # Initialize session state variables
    if "model_image" not in st.session_state:
        st.session_state.model_image = None
    if "jewelry_output_image" not in st.session_state:
        st.session_state.jewelry_output_image = None
    if "necklace_image" not in st.session_state:
        st.session_state.necklace_image = None
    if "transferred_image" not in st.session_state:
        st.session_state.transferred_image = None

    # Predefined sample images
    sample_images = [
        "model1.jpg", "model2.jpg", "model3.jpg",
        "model4.jpg", "model5.jpg", "model6.jpg"
    ]

    st.subheader("Choose a Model Image")
    selected_sample = None

    # Display sample images in a grid (2 rows × 3 columns)
    for row in range(2):
        cols = st.columns(3)
        for col_idx, col in enumerate(cols):
            sample_idx = row * 3 + col_idx
            if sample_idx < len(sample_images):
                with col:
                    st.image(sample_images[sample_idx], caption=f"Sample {sample_idx + 1}", width=150)
                    if st.button(f"Select Sample {sample_idx + 1}", key=f"sample_{sample_idx + 1}"):
                        selected_sample = sample_images[sample_idx]
                        st.session_state.model_image = cv2.imread(selected_sample)

    st.subheader("Or Upload a Model Image")
    model_image_file = st.file_uploader("Upload Model Image", type=["jpg", "jpeg", "png"])

    if model_image_file:
        st.session_state.model_image = cv2.imdecode(np.frombuffer(model_image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if st.session_state.model_image is not None:
        st.image(st.session_state.model_image, caption="Selected Model Image", channels="BGR", width=300)
    else:
        st.warning("Please select a sample image or upload your own.")
        return

    # Dropdown selection for mode
    mode = st.selectbox("Select Mode", ["Manual", "Auto"])

    # Upload necklace image
    necklace_image_file = st.file_uploader("Upload Necklace Image (PNG with transparency)")

    if necklace_image_file is not None and st.session_state.model_image is not None:
        if st.session_state.necklace_image != necklace_image_file:
            st.session_state.necklace_image = necklace_image_file
            try:
                if mode == "Manual":
                    st.session_state.jewelry_output_image = detect_neck_and_adjust_necklaces(
                        st.session_state.model_image, st.session_state.necklace_image
                    )
                else:  # Auto mode
                    st.session_state.jewelry_output_image = detect_neck_and_adjust_necklace(
                        st.session_state.model_image, st.session_state.necklace_image
                    )
            except ValueError as e:
                st.error(f"Error in jewelry try-on: {e}")

    if st.session_state.jewelry_output_image is not None:
        st.image(st.session_state.jewelry_output_image, caption="Necklace Applied", channels="BGR", width=300)

    # Makeup transfer section
    st.header("Makeup Transfer")
    reference_image_file = st.file_uploader("Upload Reference Image for Makeup", type=["jpg", "jpeg", "png"])

    if st.session_state.jewelry_output_image is not None:
        jewelry_output_image_rgb = cv2.cvtColor(st.session_state.jewelry_output_image, cv2.COLOR_BGR2RGB)
        source_image = Image.fromarray(jewelry_output_image_rgb).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.image(source_image, caption="Source Image (Jewelry Try-On Output)", width=300)
        if reference_image_file:
            reference_image = Image.open(reference_image_file).convert("RGB")
            #st.session_state.reference_image = cv2.imdecode(np.frombuffer(reference_image_file.read(), np.uint8),cv2.IMREAD_COLOR)
            #reference_image=st.session_state.reference_image
            with col2:
                st.image(reference_image, caption="Reference Image", width=300)
                #st.image(st.session_state.reference_image, caption="Selected Reference Image", channels="BGR", width=300)

            # Assuming transfer_makeup() function is available
            transferred_image = transfer_makeup(source_image, reference_image)
            if transferred_image is not None:
                st.image(transferred_image, caption="Transferred Makeup", width=300)
                st.session_state.transferred_image = transferred_image
    else:
        st.warning("Please complete the jewelry try-on section first.")

    # Fine-Tune Makeup Section
    st.sidebar.header("Fine-Tune Makeup")

    lipstick = st.sidebar.checkbox("Lipstick")
    lipstick_color = st.sidebar.color_picker("Lipstick Color", "#FF0000")
    lipstick_color = hex_to_rgb(lipstick_color)

    saturation = st.sidebar.checkbox("Saturation")
    saturation_level = st.sidebar.slider("Saturation Level", min_value=0, max_value=100, value=50)

    concealer = st.sidebar.checkbox("Concealer")
    concealer_color = st.sidebar.color_picker("Concealer Color", "#834E0D")
    concealer_color = hex_to_rgb(concealer_color)
    concealer_n = 1
    if concealer:
        status = st.radio("Select Place to apply Concealer:", ('Chin', 'Nose', 'Under Eye'))
        if status == 'Chin':
            concealer_n = 1
        if status == 'Nose':
            concealer_n = 2
        if status == 'Under Eye':
            concealer_n = 3

    undereye = st.sidebar.checkbox("Undereye")
    # if undereye:
    eye_liner_color = st.sidebar.color_picker("Undereye", "#000000")
    eye_liner_color = hex_to_rgb(eye_liner_color)

    blush = st.sidebar.checkbox("Blush")
    # if blush:
    blush_color = st.sidebar.color_picker("Blush Color", "#AA0D70")
    blush_color = hex_to_rgb(blush_color)

    lipliner = st.sidebar.checkbox("Lipliner")
    # if lipliner:
    lipliner_color = st.sidebar.color_picker("Lipliner Color", "#874511")
    lipliner_color = hex_to_rgb(lipliner_color)

    eyebrows = st.sidebar.checkbox("Eyebrows")
    # if eyebrows:
    eyebrow_color = st.sidebar.color_picker("Eyebrow Color", "#000000")
    eyebrow_color = hex_to_rgb(eyebrow_color)

    toner = st.sidebar.checkbox("Skin Toner")
    # if toner:
    toner_color = st.sidebar.color_picker("Skin Toner Color", "#A4620A")
    toner_color = hex_to_rgb(toner_color)

    eyeshadow = st.sidebar.checkbox("Eye Shadow")
    eyeshadow_color = st.sidebar.color_picker("Eye Shadow Color", "#A00E10")
    eyeshadow_color = hex_to_rgb(eyeshadow_color)

    if st.button("Apply Fine-Tune Makeup"):
        if st.session_state.jewelry_output_image is not None:
            results = face_mesh.process(cv2.cvtColor(st.session_state.jewelry_output_image, cv2.COLOR_BGR2RGB))
            # st.image(transferred_image, caption="not tuned Makeup", width=300)
            fine_tuned_image = apply_fine_tune_makeup(
                st.session_state.transferred_image, results,
                lipstick, lipstick_color, undereye, eye_liner_color,
                concealer, concealer_color, concealer_n, blush, blush_color,
                toner, toner_color, eyebrows, lipliner, lipliner_color, eyebrow_color,
                eyeshadow, eyeshadow_color, saturation, saturation_level
            )
            # final_image = Image.open(fine_tuned_image).convert("RGB")
            # fine_tuned_image = cv2.cvtColor(fine_tuned_image, cv2.COLOR_BGR2RGB)
            st.image(fine_tuned_image, caption="Fine-Tuned Makeup", width=300)
        else:
            st.warning("Make sure you have completed the previous steps first.")


if __name__ == "__main__":
    main()
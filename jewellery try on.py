import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)


# Function to remove background and add transparency
def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_white = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    _, mask_black = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    combined_mask = cv2.bitwise_or(mask_white, mask_black)
    mask_inv = cv2.bitwise_not(combined_mask)
    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    result = cv2.merge([image, mask_inv[:, :, 0]])
    return result


# Function to create a mask for the neck area
def create_neck_mask(image, landmarks, h, w):
    jaw_indices = [152, 377, 378, 379, 365, 367, 164, 0, 17, 18, 200, 199, 175, 152]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in jaw_indices]

    # Debug: Print and visualize the jawline points
    print("Jawline points (x, y):")
    for point in points:
        print(point)
        cv2.circle(image, point, 2, (0, 255, 0), -1)  # Draw green dots for jawline points

    cv2.fillPoly(mask, [np.array(points, np.int32)], (255, 255, 255))
    return mask


# Function to detect neck and adjust necklace position
def detect_neck_and_adjust_necklace(model_image_path, necklace_image_path, output_image_path):
    image = cv2.imread(model_image_path)
    h, w, _ = image.shape
    print ("image height and width ",h,w)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        #Debug: Print landmark coordinates
        print("Detected face landmarks:")
        for i, landmark in enumerate(landmarks):
            print(f"Landmark {i}: ({landmark.x * w}, {landmark.y * h})")

        # Calculate neck width and y_offset
        chin_index = 152
        left_jaw_index = 234
        right_jaw_index = 454

        chin_x = int(landmarks[chin_index].x * w)
        chin_y = int(landmarks[chin_index].y * h)
        left_jaw_x = int(landmarks[left_jaw_index].x * w)
        right_jaw_x = int(landmarks[right_jaw_index].x * w)
        neck_width = abs(right_jaw_x - left_jaw_x)
        y_offset = chin_y + 10  #necklace positioning

        # Debug: Print calculated chin and jaw coordinates
        print(f"Chin position: ({chin_x}, {chin_y})")
        print(f"Left jaw position: ({left_jaw_x}, {chin_y})")
        print(f"Right jaw position: ({right_jaw_x}, {chin_y})")
        print(f"Calculated neck width: {neck_width}")
        #print(f"Y offset for necklace: {y_offset}")

        # Draw key points on the image for visualization
        cv2.circle(image, (chin_x, chin_y), 3, (255, 0, 0), -1)  # Red dot for chin
        cv2.circle(image, (left_jaw_x, chin_y), 3, (255, 0, 0), -1)  # Red dot for left jaw
        cv2.circle(image, (right_jaw_x, chin_y), 3, (255, 0, 0), -1)  # Red dot for right jaw

        # Load and adjust the necklace
        necklace = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)

        # Print the original dimensions of the necklace before resizing
        original_height, original_width = necklace.shape[:2]
        print(f"Original necklace dimensions: width={original_width}, height={original_height}")

        necklace = remove_background(necklace)
        necklace_resized = cv2.resize(necklace, (neck_width, int(neck_width * necklace.shape[0] / necklace.shape[1])))

        # Debug: Print resized necklace dimensions
        nh, nw, _ = necklace_resized.shape
        print(f"Resized necklace dimensions: width={nw}, height={nh}")

        # Calculate x_offset to center the necklace
        x_offset = (w - nw) // 2

        # Create a mask for the neck region
        neck_mask = create_neck_mask(image, landmarks, h, w)

        # Overlay the necklace onto the image
        overlay_image_with_mask(image, necklace_resized, neck_mask, x_offset, y_offset)

        # Save the output image with visual markers
        cv2.imwrite(output_image_path, image)
        print(f"Output saved to {output_image_path}")
    else:
        print("No face detected in the image.")


# Helper function to overlay an image with transparency and masking
def overlay_image_with_mask(background, overlay, mask, x_offset, y_offset):
    bh, bw, _ = background.shape
    h, w, _ = overlay.shape

    for i in range(h):
        for j in range(w):
            if x_offset + j >= bw or y_offset + i >= bh:
                continue
            alpha = overlay[i, j, 3] / 255.0
            if alpha > 0 and mask[y_offset + i, x_offset + j] == 0:
                background[y_offset + i, x_offset + j] = (1. - alpha) * background[
                    y_offset + i, x_offset + j] + alpha * overlay[i, j, :3]



model_image_path = r'C:\Users\saran\PycharmProjects\Jewellery-Try-On\model1.jpg'  # Update with the correct path
necklace_image_path = 'neck1.jpg'  # Update with the correct path
output_image_path = 'output_with_debug.jpg'

detect_neck_and_adjust_necklace(model_image_path, necklace_image_path, output_image_path)

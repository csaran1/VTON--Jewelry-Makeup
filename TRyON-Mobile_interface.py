import sys
import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import mediapipe as mp
import time
from io import BytesIO

# Set page config FIRST
st.set_page_config(
    page_title="Mobile Jewelry Try-On",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add EleGANt to the system path
elegant_path = r'D:\try on\EleGANt'
sys.path.append(elegant_path)

# Import EleGANt modules
try:
    from EleGANt.training.config import get_config
    from EleGANt.training.inference import Inference

    ELEGANT_AVAILABLE = True
except ImportError:
    ELEGANT_AVAILABLE = False
    st.warning("EleGANt not available. Makeup transfer will be disabled.")

# MediaPipe setup for face landmarks
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize EleGANt configuration
if ELEGANT_AVAILABLE:
    class Args:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = r'D:\try on\EleGANt\ckpts\sow_pyramid_a5_e3d2_remapped.pth'
        save_folder = ''


    args = Args()
    config = get_config()

# Clean and simple CSS
st.markdown("""
<style>
    /* Remove default padding and margins */
    .stApp, .stApp > div, .stApp > div > div {
        padding: 0 !important;
        margin: 0 !important;
    }

    body, html {
        margin: 0;
        padding: 0;
    }

    /* Main image display - more compact */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 5px 0;
        max-width: 100%;
    }

    /* Force mobile images to be larger - very aggressive approach */
    @media (max-width: 768px) {
        /* Target all image containers */
        div[data-testid="stImage"], 
        .stImage,
        .main-image-container .stImage,
        .main-image-container div[data-testid="stImage"] {
            width: 100vw !important;
            max-width: 100vw !important;
            margin-left: calc(-50vw + 50%) !important;
            overflow: visible !important;
        }

        /* Target the actual img elements */
        div[data-testid="stImage"] img,
        .stImage img,
        .main-image-container img,
        .main-image-container .stImage img {
            width: 100vw !important;
            max-width: 100vw !important;
            height: auto !important;
            object-fit: contain !important;
        }

        /* Override column constraints */
        .main-image-container .stColumns > div {
            overflow: visible !important;
            width: 100% !important;
        }

        /* Make container full width */
        .main-image-container {
            width: 100vw !important;
            margin-left: calc(-50vw + 50%) !important;
            overflow: visible !important;
        }
    }

    /* Horizontal scrolling for columns */
    .stColumns {
        display: flex !important;
        overflow-x: auto !important;
        gap: 1px !important;
        padding: 2px 0 !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: thin !important;
        scrollbar-color: #4A90E2 #f1f1f1 !important;
    }

    .stColumns::-webkit-scrollbar {
        height: 4px !important;
    }

    .stColumns::-webkit-scrollbar-track {
        background: #f1f1f1 !important;
        border-radius: 2px !important;
    }

    .stColumns::-webkit-scrollbar-thumb {
        background: #4A90E2 !important;
        border-radius: 2px !important;
    }

    /* Button styles */
    .stButton > button {
        background: linear-gradient(45deg, #4A90E2, #357ABD);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 12px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
    }

    /* Mobile optimizations */
    @media (max-width: 768px) {
        .stColumns {
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            -ms-overflow-style: none;
        }

        .stColumns::-webkit-scrollbar {
            display: none;
        }
    }

    /* Style M, N, and T buttons - more compact */
    button[data-testid*="model_btn_"], button[data-testid*="necklace_btn_"], button[data-testid*="makeup_btn_"] {
        background: rgba(74, 144, 226, 0.8) !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-size: 9px !important;
        cursor: pointer !important;
        width: 100% !important;
        margin: 1px 0 !important;
        min-height: 20px !important;
    }

    /* Compact subheaders */
    .stSubheader {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.1rem !important;
    }

    /* Compact info boxes */
    .stAlert {
        margin: 0.2rem 0 !important;
        padding: 0.3rem 0.5rem !important;
    }

    button[data-testid*="model_btn_"]:hover, button[data-testid*="necklace_btn_"]:hover, button[data-testid*="makeup_btn_"]:hover {
        background: rgba(74, 144, 226, 1.0) !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Special styling for makeup transfer buttons */
    button[data-testid*="makeup_btn_"] {
        background: rgba(255, 105, 180, 0.8) !important; /* Pink color for makeup */
    }

    button[data-testid*="makeup_btn_"]:hover {
        background: rgba(255, 105, 180, 1.0) !important;
    }
</style>
""", unsafe_allow_html=True)


def segment_jewellery(image):
    """
    Segments jewellery from an image using K-Means clustering.

    Parameters:
    - image (numpy array): Input image array.

    Returns:
    - image_rgb (numpy array): Original image in RGB format.
    - mask (numpy array): Binary mask where jewellery is separated.
    - segmented (numpy array): Jewellery image with transparent background.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    h, w, _ = image.shape
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
    return image_rgb, mask, segmented


def find_necklace_3point(mask):
    """Find three key points of necklace from mask for precise positioning."""
    # Find first non-zero pixel (top-left)
    first_non_zero = None
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] != 0:
                first_non_zero = (x, y)
                break
        if first_non_zero:
            break

    # Find second non-zero pixel (top-right)
    second_non_zero = None
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1] - 1, 0, -1):
            if mask[y, x] != 0:
                second_non_zero = (x, y)
                break
        if second_non_zero:
            break

    # Find third non-zero pixel (bottom)
    third_non_zero = None
    for y in range(mask.shape[0] - 1, 0, -1):
        for x in range(mask.shape[1]):
            if mask[y, x] != 0:
                third_non_zero = (x, y)
                break
        if third_non_zero:
            break

    return first_non_zero, second_non_zero, third_non_zero


def detect_face_landmarks(image):
    """Detect face landmarks using MediaPipe Face Mesh."""
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None


def get_neck_position(landmarks, image_shape):
    """Get optimal necklace position based on face landmarks."""
    h, w = image_shape[:2]

    # Key landmarks for necklace positioning
    chin_index = 152  # Chin tip
    left_jaw_index = 234  # Left jaw
    right_jaw_index = 454  # Right jaw
    neck_index = 10  # Neck point

    # Get landmark coordinates
    chin_x = int(landmarks[chin_index].x * w)
    chin_y = int(landmarks[chin_index].y * h)
    left_jaw_x = int(landmarks[left_jaw_index].x * w)
    right_jaw_x = int(landmarks[right_jaw_index].x * w)
    neck_x = int(landmarks[neck_index].x * w)
    neck_y = int(landmarks[neck_index].y * h)

    # Calculate neck width and position
    neck_width = abs(right_jaw_x - left_jaw_x)

    # Position necklace slightly below chin, centered
    x_offset = chin_x - neck_width // 2
    y_offset = chin_y + int(0.02 * h)  # Small offset below chin

    # Ensure necklace fits within image bounds
    x_offset = max(0, min(x_offset, w - neck_width))
    y_offset = max(0, min(y_offset, h - int(neck_width * 0.3)))

    return x_offset, y_offset, neck_width


def apply_necklace_with_mediapipe(image, necklace):
    """Apply necklace with MediaPipe face detection for accurate positioning."""
    landmarks = detect_face_landmarks(image)

    if landmarks is None:
        # Fallback to simple positioning if no face detected
        h, w, _ = image.shape
        necklace_resized = cv2.resize(necklace, (w // 3, h // 6))
        x_offset = (w - necklace_resized.shape[1]) // 2
        y_offset = h // 3
        overlay_image_with_mask(image, necklace_resized, x_offset, y_offset)
        return image

    # Enhanced positioning using improved algorithm
    h, w, _ = image.shape

    # Neck-related landmark indices
    chin_index = 152  # Chin tip
    left_jaw_index = 234  # Left jaw
    right_jaw_index = 454  # Right jaw

    # Calculate neck landmark coordinates
    chin_x = int(landmarks[chin_index].x * w)
    chin_y = int(landmarks[chin_index].y * h)
    left_jaw_x = int(landmarks[left_jaw_index].x * w)
    right_jaw_x = int(landmarks[right_jaw_index].x * w)

    # Calculate neck width
    neck_width = abs(right_jaw_x - left_jaw_x)

    # Get necklace dimensions from mask if available
    try:
        # Try to get necklace dimensions from segmentation
        original, mask, _ = segment_jewellery(necklace)
        necklace_point1, necklace_point2, necklace_point3 = find_necklace_3point(mask)
        necklace_width = necklace_point2[0] - necklace_point1[0]

        # Improved scaling ratio
        ratio = (neck_width / necklace_width) * 0.95
        necklace_resized = cv2.resize(necklace, (int(necklace.shape[1] * ratio), int(necklace.shape[0] * ratio)))
    except:
        # Fallback to simple scaling
        necklace_height = int(neck_width * necklace.shape[0] / necklace.shape[1])
        necklace_resized = cv2.resize(necklace, (neck_width, necklace_height))

    # Improved Y-offset calculation
    y_offset = chin_y + int(0.25 * neck_width)
    x_offset = chin_x - necklace_resized.shape[1] // 2

    # Apply necklace with transparency
    overlay_image_with_mask(image, necklace_resized, x_offset, y_offset)
    return image


def overlay_image_with_mask(background, overlay, x_offset, y_offset):
    """Overlay image with transparency handling."""
    h, w, c = overlay.shape
    for i in range(h):
        for j in range(w):
            if x_offset + j >= background.shape[1] or y_offset + i >= background.shape[0]:
                continue
            alpha = overlay[i, j, 3] / 255.0
            if alpha > 0:
                background[y_offset + i, x_offset + j] = (
                        (1 - alpha) * background[y_offset + i, x_offset + j] + alpha * overlay[i, j, :3]
                )


def transfer_makeup(source_image, reference_image):
    """Transfer makeup from reference image to source image using EleGANt."""
    if not ELEGANT_AVAILABLE:
        st.error("EleGANt not available. Makeup transfer is disabled.")
        return None

    try:
        # Initialize the Inference object
        inference = Inference(config=config, args=args, model_path=args.model_path)

        src_img_np = np.array(source_image)
        ref_img_np = np.array(reference_image)

        # Resize reference image to match source image dimensions
        ref_img_resized_np = cv2.resize(ref_img_np, (src_img_np.shape[1], src_img_np.shape[0]))

        src_img_pil = Image.fromarray(src_img_np)
        ref_img_resized_pil = Image.fromarray(ref_img_resized_np)

        if hasattr(inference, 'transfer'):
            output_img_pil = inference.transfer(src_img_pil, ref_img_resized_pil)
            output_img_np = np.array(output_img_pil)

            if output_img_np.shape != src_img_np.shape:
                # Resize the output image to match the source image dimensions
                output_img_np_resized = cv2.resize(output_img_np, (src_img_np.shape[1], src_img_np.shape[0]))
                return output_img_np_resized
            else:
                return output_img_np
        else:
            st.error("Transfer method not found in Inference module.")
            return None
    except Exception as e:
        st.error(f"Error in makeup transfer: {e}")
        return None


def load_necklace_samples():
    """Load necklace sample images."""
    necklace_samples = []
    sample_paths = [
        r"D:\try on\Necklace Images\neck1.jpg",r"D:\try on\Necklace Images\neck2.jpg",
        r"D:\try on\Necklace Images\neck3.jpg",r"D:\try on\Necklace Images\neck4.png",
        r"D:\try on\Necklace Images\neck5.jpg", r"D:\try on\Necklace Images\neck6.png", r"D:\try on\Necklace Images\neck7.png"
    ]

    for i, path in enumerate(sample_paths):
        if os.path.exists(path):
            try:
                necklace = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if necklace is not None:
                    original, mask, segmented = segment_jewellery(necklace)
                    necklace_samples.append({
                        'name': f'N{i + 1}',
                        'path': path,
                        'image': segmented,
                        'original': original
                    })
            except Exception as e:
                st.warning(f"Could not load necklace {i + 1}: {e}")

    return necklace_samples


def load_makeup_samples():
    """Load makeup reference images."""
    makeup_samples = []
    sample_paths = [
        "Makeup1.jpg", "Makeup2.png", "Makeup3.png", "Makeup4.png",
        "Makeup5.png", "Makeup6.png", "Makeup7-a.png"
    ]

    # Fallback to some common image files if makeup-specific ones don't exist
    fallback_paths = [
        "model1.jpg", "model2.jpg", "model3.jpg", "model4.jpg",
        "model5.jpg", "model6.jpg"
    ]

    # Try makeup-specific images first
    for i, path in enumerate(sample_paths):
        if os.path.exists(path):
            try:
                makeup_img = cv2.imread(path)
                if makeup_img is not None:
                    makeup_samples.append({
                        'name': f'Makeup {i + 1}',
                        'path': path,
                        'image': makeup_img
                    })
                    print(f"Successfully loaded makeup reference: {path}")
                else:
                    print(f"Failed to load image: {path}")
            except Exception as e:
                st.warning(f"Could not load makeup reference {i + 1}: {e}")
                print(f"Error loading {path}: {e}")
        else:
            print(f"File not found: {path}")

    # If no makeup-specific images found, use fallback images
    if not makeup_samples:
        for i, path in enumerate(fallback_paths):
            if os.path.exists(path):
                try:
                    makeup_img = cv2.imread(path)
                    if makeup_img is not None:
                        makeup_samples.append({
                            'name': f'Style {i + 1}',
                            'path': path,
                            'image': makeup_img
                        })
                except Exception as e:
                    st.warning(f"Could not load fallback makeup reference {i + 1}: {e}")

    return makeup_samples


# Profile avatars function removed as not needed

def main():
    # Initialize session state
    if "current_model" not in st.session_state:
        st.session_state.current_model = 0
    if "model_images" not in st.session_state:
        st.session_state.model_images = []
    if "jewelry_output_image" not in st.session_state:
        st.session_state.jewelry_output_image = None
    if "selected_necklace" not in st.session_state:
        st.session_state.selected_necklace = None
    if "necklace_samples" not in st.session_state:
        st.session_state.necklace_samples = load_necklace_samples()
    if "makeup_samples" not in st.session_state:
        st.session_state.makeup_samples = load_makeup_samples()
    if "selected_makeup" not in st.session_state:
        st.session_state.selected_makeup = None
    if "makeup_output_image" not in st.session_state:
        st.session_state.makeup_output_image = None
    # Profile avatars removed as not needed
    if "camera_mode" not in st.session_state:
        st.session_state.camera_mode = False

    # Load model images
    model_paths = ["model1.jpg", "model2.jpg", "model3.jpg", "model4.jpg", "model5.jpg", "model6.jpg"]
    if not st.session_state.model_images:
        for path in model_paths:
            if os.path.exists(path):
                img = cv2.imread(path)
                st.session_state.model_images.append(img)

    # Main image display - Show makeup result, jewelry result, or original model (smaller size, centered)
    if st.session_state.makeup_output_image is not None:
        # Check if mobile using JavaScript detection
        st.markdown("""
        <script>
        if (window.innerWidth <= 768) {
            document.body.classList.add('mobile-view');
        }
        </script>
        """, unsafe_allow_html=True)

        st.markdown('<div class="main-image-container">', unsafe_allow_html=True)
        # On mobile, use full width; on desktop, use columns
        col1, col2, col3 = st.columns([0.1, 1, 0.1])  # Narrower side columns
        with col2:
            st.image(st.session_state.makeup_output_image, channels="BGR", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.jewelry_output_image is not None:
        st.markdown('<div class="main-image-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.1, 1, 0.1])
        with col2:
            st.image(st.session_state.jewelry_output_image, channels="BGR", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.model_images:
        st.markdown('<div class="main-image-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.1, 1, 0.1])
        with col2:
            current_img = st.session_state.model_images[st.session_state.current_model]
            st.image(current_img, channels="BGR", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload an image to get started")

    # Upload functionality
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"], key="uploader")
    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.session_state.model_images.append(img)
        st.session_state.current_model = len(st.session_state.model_images) - 1
        st.rerun()

    # Model selection - Horizontal scrolling with compact layout
    if st.session_state.model_images:
        st.subheader("Select Model")

        # Create horizontal scrolling columns
        model_cols = st.columns(len(st.session_state.model_images[:6]))
        for i, img in enumerate(st.session_state.model_images[:6]):
            with model_cols[i]:
                # Display image and button in same column - Fix color channels (smaller size)
                st.image(img, width=50, caption=f"M{i + 1}", channels="BGR")
                if st.button(f"M{i + 1}", key=f"model_btn_{i}", help=f"Select Model {i + 1}"):
                    st.session_state.current_model = i
                    st.rerun()

    # Necklace selection
    if st.session_state.necklace_samples:
        st.subheader("💎 Select a Necklace")

        # Create horizontal scrolling columns
        necklace_cols = st.columns(len(st.session_state.necklace_samples))
        for i, necklace in enumerate(st.session_state.necklace_samples):
            with necklace_cols[i]:
                # Display image and button in same column - Fix necklace color channels (smaller size)
                # Convert RGBA to RGB for display
                necklace_img = necklace['image']
                if len(necklace_img.shape) == 3 and necklace_img.shape[2] == 4:  # RGBA
                    # Convert BGRA to BGR for Streamlit
                    necklace_display = necklace_img[:, :, :3]  # Remove alpha channel
                else:
                    necklace_display = necklace_img

                st.image(necklace_display, width=50, caption=f"N{i + 1}", channels="BGR")
                if st.button(f"N{i + 1}", key=f"necklace_btn_{i}", help=f"Try on {necklace['name']}"):
                    st.session_state.selected_necklace = necklace

                    if st.session_state.model_images:
                        try:
                            result_img = st.session_state.model_images[st.session_state.current_model].copy()
                            result_img = apply_necklace_with_mediapipe(result_img, necklace['image'])
                            st.session_state.jewelry_output_image = result_img
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error applying necklace: {e}")

    # Makeup transfer selection - Horizontal scrolling with compact layout
    if st.session_state.makeup_samples:
        st.subheader("Transfer Makeup Style")

        # Create horizontal scrolling columns
        makeup_cols = st.columns(len(st.session_state.makeup_samples))
        for i, makeup in enumerate(st.session_state.makeup_samples):
            with makeup_cols[i]:
                # Display image and button in same column - Fix makeup color channels (smaller size)
                st.image(makeup['image'], width=50, caption=f"T{i + 1}", channels="BGR")
                if st.button(f"T{i + 1}", key=f"makeup_btn_{i}", help=f"Transfer {makeup['name']}"):
                    st.session_state.selected_makeup = makeup

                    if st.session_state.model_images:
                        try:
                            # Always use jewelry result as input if available, never use previous makeup result
                            if st.session_state.jewelry_output_image is not None:
                                # Use jewelry result as input for makeup transfer (fresh base)
                                source_img_rgb = cv2.cvtColor(st.session_state.jewelry_output_image, cv2.COLOR_BGR2RGB)
                                source_type = "jewelry result"
                            else:
                                # Use original model as input
                                source_img_rgb = cv2.cvtColor(
                                    st.session_state.model_images[st.session_state.current_model], cv2.COLOR_BGR2RGB)
                                source_type = "original model"

                            makeup_img_rgb = cv2.cvtColor(makeup['image'], cv2.COLOR_BGR2RGB)

                            # Apply makeup transfer
                            result_img_rgb = transfer_makeup(source_img_rgb, makeup_img_rgb)

                            if result_img_rgb is not None:
                                # Convert back to BGR for display
                                result_img_bgr = cv2.cvtColor(result_img_rgb, cv2.COLOR_RGB2BGR)
                                st.session_state.makeup_output_image = result_img_bgr
                                st.rerun()
                            else:
                                st.error("Makeup transfer failed!")
                        except Exception as e:
                            st.error(f"Error applying makeup transfer: {e}")

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reset"):
            st.session_state.jewelry_output_image = None
            st.session_state.selected_necklace = None
            st.session_state.makeup_output_image = None
            st.session_state.selected_makeup = None
            st.rerun()
    with col2:
        if st.button("💾 Save"):
            if st.session_state.makeup_output_image is not None:
                cv2.imwrite(f"makeup_output_{int(time.time())}.jpg", st.session_state.makeup_output_image)
                st.success("Makeup result saved successfully!")
            elif st.session_state.jewelry_output_image is not None:
                cv2.imwrite(f"jewelry_output_{int(time.time())}.jpg", st.session_state.jewelry_output_image)
                st.success("Jewelry result saved successfully!")
            else:
                st.warning("No processed image to save!")


if __name__ == "__main__":
    main()

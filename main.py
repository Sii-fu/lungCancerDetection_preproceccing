import cv2
import numpy as np
import os
from tkinter import Tk, Button, Label, Frame, filedialog, messagebox
from PIL import Image, ImageTk

# --- Functions ---

# Global variables to store image references and processing state
current_images = {}
current_img_path = None
processing_step = 0
original_img = None
pipeline_results = {}  # stores intermediate results for each pipeline step


def preprocess_pipeline(img_bgr):
    """
    Run the preprocessing pipeline and return a dict of step name -> image
    Steps:
      Original (RGB), CLAHE (preserve color via YCrCb), Resized (300x300),
      Seg+ROI (grayscale ROI padded/resized), Normalized (0-1 float), Heatmap (RGB)
    """
    results = {}
    # Step 0: Original (RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results["Original"] = img_rgb

    # Step 1: CLAHE on Y channel (preserve color)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y.astype(np.uint8))
    ycrcb = cv2.merge((y, cr, cb))
    img_clahe = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    results["CLAHE"] = img_clahe

    # Step 2: Resize to 300x300
    resized = cv2.resize(img_clahe, (300, 300))
    results["Resized"] = resized

    # Step 3: Segmentation + ROI crop using an IPCT-like clustering method
    gray_for_seg = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    def ipct_segmentation(gray):
        """IPCT-like segmentation: spatial+intensity k-means, morphological refinement.
        Returns (segmented_gray, final_mask)
        """
        # Denoise with Non-local Means first
        den = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        h_img, w_img = den.shape[:2]

        # Build feature vector: intensity + normalized spatial coordinates
        intens = den.reshape(-1, 1).astype(np.float32)
        xs, ys = np.meshgrid(np.arange(w_img), np.arange(h_img))
        xs = (xs.reshape(-1, 1) / float(w_img) * 50).astype(np.float32)
        ys = (ys.reshape(-1, 1) / float(h_img) * 50).astype(np.float32)
        feats = np.hstack([intens, xs, ys])

        # K-means clustering (k=3) - approximate IPCT multi-cluster step
        k = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        attempts = 3
        # create an initial labels array (required by type checkers / cv2.kmeans signature)
        best_labels = np.zeros((feats.shape[0], 1), dtype=np.int32)
        _, labels, centers = cv2.kmeans(feats, k, best_labels, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()

        # Choose cluster likely representing lung region (darker cluster with sufficient area)
        centers_intensity = centers[:, 0]
        order = np.argsort(centers_intensity)
        chosen = None
        for idx in order:
            area = np.sum(labels == idx)
            if area > 0.005 * h_img * w_img:
                chosen = idx
                break
        if chosen is None:
            chosen = order[0]

        mask = (labels == chosen).astype(np.uint8).reshape(h_img, w_img) * 255

        # Morphological refinement
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Fill holes using floodFill
        im_floodfill = mask.copy()
        h2, w2 = mask.shape
        ff_mask = np.zeros((h2 + 2, w2 + 2), np.uint8)
        cv2.floodFill(im_floodfill, ff_mask, (0, 0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        mask_filled = mask | im_floodfill_inv

        # Keep only largest connected component
        contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            final_mask = np.zeros_like(mask_filled)
            cv2.drawContours(final_mask, [largest], -1, 255, -1)
        else:
            final_mask = mask_filled

        segmented = cv2.bitwise_and(gray, final_mask)
        return segmented, final_mask

    segmented, seg_mask = ipct_segmentation(gray_for_seg)

    # Crop ROI from segmented image using contours on the refined mask
    contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray_for_seg)
    roi = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        x, y, w_box, h_box = cv2.boundingRect(largest)
        roi = segmented[y:y+h_box, x:x+w_box]
        if roi.size == 0:
            roi = None
    else:
        segmented = gray_for_seg

    if roi is None:
        roi_resized = cv2.resize(segmented, (300, 300))
    else:
        # make square with padding before resize
        h_roi, w_roi = roi.shape[:2]
        size = max(h_roi, w_roi)
        padded = np.zeros((size, size), dtype=roi.dtype)
        padded[:h_roi, :w_roi] = roi
        roi_resized = cv2.resize(padded, (300, 300))

    results["Seg+ROI"] = roi_resized

    # Step 4: Normalization (0-1)
    norm_img = roi_resized.astype(np.float32) / 255.0
    results["Normalized"] = norm_img

    # Step 5: Heatmap (visualization only)
    heatmap = cv2.applyColorMap((roi_resized).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    results["Heatmap"] = heatmap

    return results

def load_and_display_original(img_path):
    """Load and display the original image"""
    global original_img, current_img_path, processing_step

    try:
        # Read original image
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image!")
            return False

        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img_rgb
        current_img_path = img_path
        processing_step = 0
        pipeline_results.clear()

        # Resize for display (maintain aspect ratio)
        height, width = img_rgb.shape[:2]
        display_size = 420
        if width > height:
            new_width = display_size
            new_height = int(height * display_size / width)
        else:
            new_height = display_size
            new_width = int(width * display_size / height)

        resized_for_display = cv2.resize(img_rgb, (new_width, new_height))

        # Convert to ImageTk
        orig_img_tk = ImageTk.PhotoImage(Image.fromarray(resized_for_display))

        # Store reference and update display
        current_images['orig'] = orig_img_tk
        label_orig.config(image=orig_img_tk, text="")
        label_processed.config(image="", text="Select 'Next Step' to start processing")

        # Update status and enable next button
        status_label.config(text=f"Loaded: {os.path.basename(img_path)} - Ready for processing")
        btn_next.config(state="normal", text="Next Step", command=process_next_step)

        return True

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")
        return False


def process_next_step():
    global processing_step, pipeline_results

    if original_img is None:
        messagebox.showwarning("Warning", "Please select an image first")
        return

    try:
        # Run the preprocess_pipeline once and store results
        if processing_step == 0:
            # load original BGR from path
            img_bgr = cv2.imread(current_img_path)
            if img_bgr is None:
                messagebox.showerror("Error", "Failed to load image for processing")
                return
            pipeline_results = preprocess_pipeline(img_bgr)

        # Map processing_step -> step name order
        step_names = ["Original", "CLAHE", "Resized", "Seg+ROI", "Normalized", "Heatmap"]
        if processing_step < len(step_names):
            step_name = step_names[processing_step]
            result_img = pipeline_results.get(step_name)
            if result_img is None:
                messagebox.showerror("Error", f"Pipeline did not produce step: {step_name}")
                return
            display_processed_image(result_img, step_name)
            processing_step += 1
            if processing_step >= len(step_names):
                btn_next.config(text="Reset", command=reset_processing)
        else:
            messagebox.showinfo("Info", "All processing steps completed!")

    except Exception as e:
        messagebox.showerror("Error", f"Processing failed: {e}")


def display_processed_image(img, step_name):
    """Display the processed image on the right side"""
    try:
        # Make display square area but preserve aspect where possible
        display_size = 420
        # Normalize and convert float images (0-1) to uint8 for display
        img_disp = img
        if isinstance(img_disp, np.ndarray) and img_disp.dtype == np.float32:
            # assume 0-1 range
            img_disp = (np.clip(img_disp, 0.0, 1.0) * 255).astype(np.uint8)

        # handle grayscale or color uniformly
        if img_disp.ndim == 2:
            h, w = img_disp.shape[:2]
            if w > h:
                new_w = display_size
                new_h = int(h * display_size / w)
            else:
                new_h = display_size
                new_w = int(w * display_size / h)
            resized_for_display = cv2.resize(img_disp, (new_w, new_h))
            canvas = 255 * np.ones((display_size, display_size, 3), dtype=np.uint8)
            y = (display_size - new_h) // 2
            x = (display_size - new_w) // 2
            canvas[y:y+new_h, x:x+new_w] = cv2.cvtColor(resized_for_display, cv2.COLOR_GRAY2RGB)
            img_rgb = canvas
        else:
            # color image (assumed RGB)
            h, w = img_disp.shape[:2]
            if w > h:
                new_w = display_size
                new_h = int(h * display_size / w)
            else:
                new_h = display_size
                new_w = int(w * display_size / h)
            resized_for_display = cv2.resize(img_disp, (new_w, new_h))
            canvas = 255 * np.ones((display_size, display_size, 3), dtype=np.uint8)
            y = (display_size - new_h) // 2
            x = (display_size - new_w) // 2
            canvas[y:y+new_h, x:x+new_w] = resized_for_display
            img_rgb = canvas

        # Convert to ImageTk
        processed_img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        # Store reference and update display
        current_images['processed'] = processed_img_tk
        label_processed.config(image=processed_img_tk, text="")

        # Update status
        status_label.config(text=f"Step {min(processing_step+1,5)}: {step_name}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to display processed image: {e}")


def reset_processing():
    """Reset the processing to start over"""
    global processing_step
    processing_step = 0
    btn_next.config(text="Next Step", command=process_next_step)
    if current_img_path:
        load_and_display_original(current_img_path)

def load_default_image():
    """Load a default test image"""
    default_path = "dataset/IQ-OTHNCCD/Bengin cases/Bengin case (1).jpg"
    if os.path.exists(default_path):
        return default_path
    return None

def select_image():
    initial_dir = "dataset/Test cases" if os.path.exists("dataset/Test cases") else "."
    path = filedialog.askopenfilename(
        initialdir=initial_dir,
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if path:
        load_and_display_original(path)

def load_default():
    """Load default test image"""
    default_path = load_default_image()
    if default_path:
        load_and_display_original(default_path)
    else:
        messagebox.showwarning("Warning", "No default test image found")

# --- UI ---
root = Tk()
root.title("Lung Segmentation Demo - Step by Step Processing")
root.geometry("900x700")

# Create a frame for buttons
button_frame = Frame(root)
button_frame.pack(pady=10)

btn_select = Button(button_frame, text="Select Image", command=select_image)
btn_select.pack(side="left", padx=5)

btn_default = Button(button_frame, text="Load Default Test Image", command=load_default)
btn_default.pack(side="left", padx=5)

btn_next = Button(button_frame, text="Next Step", command=process_next_step, state="disabled")
btn_next.pack(side="left", padx=5)

# Create a frame for images
image_frame = Frame(root)
image_frame.pack(pady=10, expand=True, fill="both")

# Left side - Original image
left_frame = Frame(image_frame)
left_frame.pack(side="left", padx=10, pady=10, expand=True, fill="both")

Label(left_frame, text="Original Image", font=("Arial", 12, "bold")).pack()
label_orig = Label(left_frame, text="No image selected", width=50, height=25, relief="sunken", bg="white")
label_orig.pack(expand=True, fill="both")

# Right side - Processed image
right_frame = Frame(image_frame)
right_frame.pack(side="right", padx=10, pady=10, expand=True, fill="both")

Label(right_frame, text="Processed Result", font=("Arial", 12, "bold")).pack()
label_processed = Label(right_frame, text="Select an image to start", width=50, height=25, relief="sunken", bg="lightgray")
label_processed.pack(expand=True, fill="both")

# Status label
status_label = Label(root, text="Ready - Select an image to begin processing", relief="sunken", anchor="w")
status_label.pack(side="bottom", fill="x", pady=5)

root.mainloop()

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
    """Process the next step in the pipeline in order:
    0: Grayscale
    1: CLAHE (histogram enhancement)
    2: Resize (300x300)
    3: Denoise (median blur)
    4: Segmentation
    """
    global processing_step, original_img

    if original_img is None:
        messagebox.showwarning("Warning", "Please select an image first")
        return

    try:
        # We'll derive each step from original_img so each step is independent
        if processing_step == 0:
            # Step 1: Convert to grayscale
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            result_img = gray
            step_name = "Grayscale Conversion"

        elif processing_step == 1:
            # Step 2: CLAHE (histogram enhancement)
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            result_img = enhanced
            step_name = "CLAHE Histogram Enhancement"

        elif processing_step == 2:
            # Step 3: Resize
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            resized = cv2.resize(enhanced, (300, 300))
            result_img = resized
            step_name = "Resized to 300x300"

        elif processing_step == 3:
            # Step 4: Denoise
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            resized = cv2.resize(enhanced, (300, 300))
            # denoised = cv2.medianBlur(resized, 3)
            denoised = cv2.GaussianBlur(resized, (5, 5), 0)
            result_img = denoised
            step_name = "Denoising Applied"

        elif processing_step == 4:
            # Step 5: Segmentation based on denoised image
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            resized = cv2.resize(enhanced, (300, 300))
            # denoised = cv2.medianBlur(resized, 3)
            denoised = cv2.GaussianBlur(resized, (5, 5), 0)
            # denoised = enhanced

            # Segmentation
            _, thresh = cv2.threshold(denoised, 100, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(denoised)
            largest = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest], -1, 255, -1)
            segmented = cv2.bitwise_and(denoised, mask)

            # --- NEW: compute tight bounding box around the lung mask and crop ROI ---
            roi = None
            if largest is not None:
                x, y, w, h = cv2.boundingRect(largest)
                # Crop from the segmented (denoised) image using bounding rect
                roi = segmented[y:y+h, x:x+w]
                # If the ROI is empty for any reason, fallback to segmented full image
                if roi.size == 0:
                    roi = None

            if roi is None:
                # fallback: use full segmented image
                roi_resized = cv2.resize(segmented, (300, 300))
                result_img = roi_resized
            else:
                # Optional padding to make square before resizing
                h_roi, w_roi = roi.shape[:2]
                size = max(h_roi, w_roi)
                padded = np.zeros((size, size), dtype=roi.dtype)
                # place roi at top-left (you can center if preferred)
                padded[:h_roi, :w_roi] = roi
                roi_resized = cv2.resize(padded, (300, 300))
                result_img = roi_resized

            step_name = "Lung Segmentation + ROI crop"

        else:
            messagebox.showinfo("Info", "All processing steps completed!")
            return

        # Display the result
        display_processed_image(result_img, step_name)
        processing_step += 1

        # If we've finished the last step, change button to Reset
        if processing_step > 4:
            btn_next.config(text="Reset", command=reset_processing)

    except Exception as e:
        messagebox.showerror("Error", f"Processing failed: {e}")


def display_processed_image(img, step_name):
    """Display the processed image on the right side"""
    try:
        # Make display square area but preserve aspect where possible
        display_size = 420
        if len(img.shape) == 2:  # Grayscale
            # Fit into display square while preserving aspect
            h, w = img.shape[:2]
            if w > h:
                new_w = display_size
                new_h = int(h * display_size / w)
            else:
                new_h = display_size
                new_w = int(w * display_size / h)
            resized_for_display = cv2.resize(img, (new_w, new_h))
            # pad to square background (white)
            canvas = 255 * np.ones((display_size, display_size, 3), dtype=np.uint8)
            y = (display_size - new_h) // 2
            x = (display_size - new_w) // 2
            canvas[y:y+new_h, x:x+new_w] = cv2.cvtColor(resized_for_display, cv2.COLOR_GRAY2RGB)
            img_rgb = canvas
        else:
            # Color image
            h, w = img.shape[:2]
            if w > h:
                new_w = display_size
                new_h = int(h * display_size / w)
            else:
                new_h = display_size
                new_w = int(w * display_size / h)
            resized_for_display = cv2.resize(img, (new_w, new_h))
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

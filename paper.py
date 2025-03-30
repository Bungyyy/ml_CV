import os
import cv2
import numpy as np
from pathlib import Path
import csv
import time
import re
import difflib
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from ultralytics import YOLO

# === Helper Functions ===
def normalize(text):
    """Normalize text by removing spaces and converting to uppercase"""
    return re.sub(r'\s+', '', text).strip().upper()

def fuzzy_match(a, b, threshold=0.85):
    """Compare two strings and determine if they're a close enough match"""
    if not a or not b:
        return 0, False
    
    # Normalize both strings for comparison by removing all non-alphanumeric chars
    a_norm = re.sub(r'[^A-Z0-9]', '', a.upper())
    b_norm = re.sub(r'[^A-Z0-9]', '', b.upper())
    
    ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio, ratio >= threshold

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

# === Enhanced Image Processing ===
def enhance_image(img, is_barcode=False):
    """Apply advanced image enhancement techniques"""
    # Convert to grayscale if it's not already
    if isinstance(img, np.ndarray) and len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif isinstance(img, Image.Image):
        gray = ImageOps.grayscale(img)
        gray = np.array(gray)
    else:
        gray = img  # Assume it's already grayscale
    
    # Apply different processing based on whether it's a barcode or text
    if is_barcode:
        # For barcodes: high contrast, thresholding, dilation to connect lines
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilate horizontally for barcodes to connect broken lines
        kernel = np.ones((1,2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        return dilated
    else:
        # For text: moderate contrast enhancement, noise reduction, sharpening
        # Apply bilateral filter for noise reduction while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(filtered)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Adaptive thresholding for better text separation
        binary = cv2.adaptiveThreshold(
            sharpened, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        return binary

# === Barcode and Three-Segment Code Detection ===
def split_image(img):
    """Split the image into potential barcode and three-segment code regions"""
    # Convert PIL to OpenCV if needed
    if isinstance(img, Image.Image):
        img = np.array(img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Define regions based on typical placement
    top_height = int(height * 0.25)  # Assume top 25% has the three-segment code
    
    # Barcode is often in the bottom portion
    barcode_regions = [(0, top_height, width, height)]
    
    # Define the three-segment code region based on typical placement
    # Usually at the top of the image
    three_segment_regions = [(0, 0, width, top_height)]
    
    return barcode_regions, three_segment_regions

# === Recognition Functions ===
def recognize_barcode(image, expected_barcode=""):
    """Extract barcode from image using OCR"""
    # Convert PIL to NumPy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Enhance the image for barcode detection
    enhanced = enhance_image(image, is_barcode=True)
    
    # Use multiple approaches to read the barcode
    decoded_barcode = ""
    ocr_results = []
    
    # Try different tesseract configurations
    try:
        # PSM 6 - Assume a single uniform block of text
        ocr_text_6 = pytesseract.image_to_string(
            enhanced, 
            config='--psm 6 -c tessedit_char_whitelist=TH0123456789'
        )
        ocr_results.append(ocr_text_6)
        
        # PSM 7 - Treat the image as a single text line
        ocr_text_7 = pytesseract.image_to_string(
            enhanced, 
            config='--psm 7 -c tessedit_char_whitelist=TH0123456789'
        )
        ocr_results.append(ocr_text_7)
        
        # Try inverting the image if we didn't get good results
        if not any("TH" in result for result in ocr_results):
            inverted = cv2.bitwise_not(enhanced)
            ocr_text_inv = pytesseract.image_to_string(
                inverted, 
                config='--psm 6 -c tessedit_char_whitelist=TH0123456789'
            )
            ocr_results.append(ocr_text_inv)
    except Exception as e:
        print(f"OCR error: {str(e)}")
    
    # Process OCR results
    for result in ocr_results:
        for line in result.splitlines():
            line_clean = normalize(line)
            
            # Typical barcode starts with TH and is followed by numbers
            if "TH" in line_clean:
                # Apply OCR corrections
                corrected = line_clean
                # Common OCR corrections
                corrections = {
                    'O': '0',  # O → 0
                    'I': '1',  # I → 1
                    'Z': '2',  # Z → 2
                    'Q': '0',  # Q → 0
                    'D': '0',  # D → 0 (similar shapes)
                }
                for bad, good in corrections.items():
                    corrected = corrected.replace(bad, good)
                
                # If we have an expected barcode, compare similarity
                if expected_barcode:
                    ratio, _ = fuzzy_match(corrected, expected_barcode, 0)
                    if ratio > 0.7:  # Accept if reasonably similar
                        decoded_barcode = corrected
                        break
                else:
                    # Without expected barcode, just take the first valid one
                    decoded_barcode = corrected
                    break
    
    # Calculate similarity
    barcode_sim = 0
    if decoded_barcode and expected_barcode:
        barcode_sim, _ = fuzzy_match(decoded_barcode, expected_barcode)
    
    return decoded_barcode, barcode_sim

def recognize_three_segment_code(image, expected_code=""):
    """Extract three-segment code from image using OCR"""
    # Convert PIL to NumPy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Enhance the image for text recognition
    enhanced = enhance_image(image, is_barcode=False)
    
    # Try multiple OCR approaches
    potential_codes = []
    
    try:
        # Try different PSM modes
        psm_modes = [6, 7, 8, 11]
        
        for psm in psm_modes:
            ocr_text = pytesseract.image_to_string(
                enhanced, 
                config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. '
            )
            
            lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
            for line in lines:
                # Look for patterns that match three-segment code
                if "-" in line or len(line) >= 8:
                    potential_codes.append(line)
                
                # Try pattern matching
                pattern = r'([A-Z0-9]{3})[\s\-_.]+([A-Z0-9]{4})[\s\-_.]+([A-Z0-9]{3})'
                match = re.search(pattern, line.upper())
                if match:
                    extracted = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                    potential_codes.append(extracted)
    except Exception as e:
        print(f"OCR error: {str(e)}")
    
    # Choose the best match from potential codes
    decoded_code = ""
    best_match_ratio = 0
    
    if expected_code:
        # If we have an expected code, compare and find best match
        for code in potential_codes:
            code_clean = normalize(code)
            ratio, _ = fuzzy_match(code_clean, expected_code, 0)
            if ratio > best_match_ratio:
                best_match_ratio = ratio
                decoded_code = code_clean
    else:
        # Without expected code, take the first one that looks like XXX-XXXX-XXX
        for code in potential_codes:
            if re.search(r'[A-Z0-9]{3}-[A-Z0-9]{4}-[A-Z0-9]{3}', code):
                decoded_code = normalize(code)
                break
        
        # If no well-formatted code found, just take the first one
        if not decoded_code and potential_codes:
            decoded_code = normalize(potential_codes[0])
    
    # Calculate similarity
    code_sim = 0
    if decoded_code and expected_code:
        code_sim, _ = fuzzy_match(decoded_code, expected_code)
    
    return decoded_code, code_sim

# === YOLO Detection Function (from yolo_detect_all.py) ===
def detect_and_save_images(input_folder, output_folder, model_path, debug_dir=None):
    """
    Detect objects in all images in a folder using YOLO and save the results.
    
    Parameters:
    - input_folder: Path to folder containing input images
    - output_folder: Path to folder for saving output images with bounding boxes
    - model_path: Path to model weights
    - debug_dir: Directory for saving debug images
    
    Returns:
    - A list of detected image paths
    - A dictionary of timing statistics
    """
    # Create output folders
    create_directory(output_folder)
    if debug_dir:
        yolo_debug_dir = os.path.join(debug_dir, "yolo_detections")
        create_directory(yolo_debug_dir)
    else:
        yolo_debug_dir = None
    
    # Timing for model loading
    model_load_start = time.time()
    # Load model
    model = YOLO(model_path)
    model_load_time = time.time() - model_load_start
    print(f"YOLO model loading time: {model_load_time:.3f}s")
    
    # Get list of image files in the input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f"*{ext}")))
        image_files.extend(list(Path(input_folder).glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return [], {"avg": 0, "max": 0, "min": 0, "model_load": model_load_time}
    
    print(f"Found {len(image_files)} images to process with YOLO")
    
    # Process each image
    detected_images = []
    processing_times = []
    detection_times = []
    saving_times = []
    detailed_timings = {}
    
    for image_path in image_files:
        image_file = os.path.basename(image_path)
        
        # Construct output paths
        output_path = os.path.join(output_folder, image_file)
        
        # Measure overall processing time
        start_time = time.time()
        
        # Measure prediction time
        predict_start = time.time()
        # Perform prediction
        results = model.predict(str(image_path))
        predict_time = time.time() - predict_start
        detection_times.append(predict_time)
        
        # Access first result
        result = results[0]
        
        # Measure saving time
        save_start = time.time()
        # Save with bounding boxes
        result.save(filename=output_path)
        save_time = time.time() - save_start
        saving_times.append(save_time)
        
        # Calculate overall processing time
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # Store detailed timing info
        detailed_timings[image_file] = {
            "total": processing_time,
            "detection": predict_time,
            "saving": save_time
        }
        
        # Also save to debug directory if specified
        if yolo_debug_dir:
            debug_path = os.path.join(yolo_debug_dir, image_file)
            result.save(filename=debug_path)
        
        detected_images.append(output_path)
        print(f"YOLO Detection: {image_file} - Total: {processing_time:.3f}s (Detection: {predict_time:.3f}s, Saving: {save_time:.3f}s)")
    
    # Calculate time statistics
    timing_stats = {}
    if processing_times:
        # Overall processing stats
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        # Detection stats
        avg_detection = sum(detection_times) / len(detection_times)
        max_detection = max(detection_times)
        min_detection = min(detection_times)
        
        # Saving stats
        avg_saving = sum(saving_times) / len(saving_times)
        max_saving = max(saving_times)
        min_saving = min(saving_times)
        
        print("\n=== YOLO Detection Time Statistics ===")
        print(f"Average processing time: {avg_time:.3f}s")
        print(f"Maximum processing time: {max_time:.3f}s")
        print(f"Minimum processing time: {min_time:.3f}s")
        
        print("\n--- Detection Time (YOLO inference) ---")
        print(f"Average detection time: {avg_detection:.3f}s")
        print(f"Maximum detection time: {max_detection:.3f}s")
        print(f"Minimum detection time: {min_detection:.3f}s")
        
        print("\n--- Saving Time ---")
        print(f"Average saving time: {avg_saving:.3f}s")
        print(f"Maximum saving time: {max_saving:.3f}s")
        print(f"Minimum saving time: {min_saving:.3f}s")
        
        timing_stats = {
            "avg": avg_time,
            "max": max_time,
            "min": min_time,
            "model_load": model_load_time,
            "detection": {
                "avg": avg_detection,
                "max": max_detection,
                "min": min_detection
            },
            "saving": {
                "avg": avg_saving,
                "max": max_saving,
                "min": min_saving
            },
            "detailed": detailed_timings
        }
    
    return detected_images, timing_stats

# === YOLO Crop Function (from yolo_crop_target.py) ===
def crop_objects(input_folder, output_folder, model_path, target_class="Target", debug_dir=None):
    """
    Process all images in a folder, detect objects labeled with the target class,
    crop them, and save to output folder.
    
    Parameters:
    - input_folder: Path to folder containing input images
    - output_folder: Path to folder for saving cropped objects
    - model_path: Path to model weights
    - target_class: Class name to detect and crop
    - debug_dir: Directory for saving debug images
    
    Returns:
    - A list of cropped image paths
    - A dictionary of timing statistics
    """
    # Create output folders
    create_directory(output_folder)
    if debug_dir:
        crop_debug_dir = os.path.join(debug_dir, "cropped_targets")
        create_directory(crop_debug_dir)
    else:
        crop_debug_dir = None
    
    # Timing for model loading
    model_load_start = time.time()
    # Load YOLOv model
    model = YOLO(model_path)
    model_load_time = time.time() - model_load_start
    print(f"YOLO model loading time (crop): {model_load_time:.3f}s")
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f"*{ext}")))
        image_files.extend(list(Path(input_folder).glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return [], {"avg": 0, "max": 0, "min": 0, "model_load": model_load_time}
    
    print(f"Found {len(image_files)} images to process for cropping")
    
    # Process each image
    target_count = 0
    cropped_images = []
    processing_times = []
    reading_times = []
    detection_times = []
    cropping_times = []
    saving_times = []
    detailed_timings = {}
    crops_per_image = {}
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"Processing for crop: {img_name}")
        
        # Measure overall processing time
        start_time = time.time()
        
        # Measure image reading time
        read_start = time.time()
        # Read the image
        img = cv2.imread(str(img_path))
        read_time = time.time() - read_start
        reading_times.append(read_time)
        
        if img is None:
            print(f"Error reading image: {img_path}")
            continue
        
        # Measure detection time
        detect_start = time.time()
        # Run inference
        results = model(img)
        detect_time = time.time() - detect_start
        detection_times.append(detect_time)
        
        # Process results
        result = results[0]  # Get the first result
        
        # Get bounding boxes, confidence scores, and class names
        boxes = result.boxes
        
        # Find all objects with the target class
        target_indices = []
        for i, cls in enumerate(boxes.cls):
            class_name = result.names[int(cls)]
            if class_name.lower() == target_class.lower():
                target_indices.append(i)
        
        if not target_indices:
            print(f"No '{target_class}' found in {img_name}")
            # Still include the processing time even if no targets found
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            crops_per_image[img_name] = 0
            detailed_timings[img_name] = {
                "total": processing_time,
                "reading": read_time,
                "detection": detect_time,
                "cropping": 0,
                "saving": 0,
                "crops": 0
            }
            continue
        
        # Measure cropping and saving time
        crop_save_start = time.time()
        crop_time_total = 0
        save_time_total = 0
        crops_count = 0
        
        # Crop and save each target object
        for idx, target_idx in enumerate(target_indices):
            # Get bounding box coordinates (x1, y1, x2, y2)
            box = boxes.xyxy[target_idx].cpu().numpy().astype(int)
            conf = float(boxes.conf[target_idx])
            
            # Ensure coordinates are within image boundaries
            height, width = img.shape[:2]
            x1, y1, x2, y2 = box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Measure crop time
            crop_start = time.time()
            # Crop the image
            cropped_img = img[y1:y2, x1:x2]
            crop_time = time.time() - crop_start
            crop_time_total += crop_time
            
            if cropped_img.size == 0:
                print(f"Warning: Empty crop in {img_name} at {box}")
                continue
            
            # Save the cropped image
            base_name = os.path.splitext(img_name)[0]
            output_name = f"{base_name}_{target_class}_{idx}_{conf:.2f}.jpg"
            output_path = os.path.join(output_folder, output_name)
            
            # Measure save time
            save_start = time.time()
            cv2.imwrite(output_path, cropped_img)
            save_time = time.time() - save_start
            save_time_total += save_time
            
            # Also save to debug directory if specified
            if crop_debug_dir:
                debug_path = os.path.join(crop_debug_dir, output_name)
                cv2.imwrite(debug_path, cropped_img)
            
            cropped_images.append(output_path)
            target_count += 1
            crops_count += 1
        
        # Add average crop and save times
        if crops_count > 0:
            cropping_times.append(crop_time_total / crops_count)
            saving_times.append(save_time_total / crops_count)
        
        # Calculate overall processing time
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # Store detailed timing info
        crops_per_image[img_name] = crops_count
        detailed_timings[img_name] = {
            "total": processing_time,
            "reading": read_time,
            "detection": detect_time,
            "cropping": crop_time_total / max(1, crops_count),
            "saving": save_time_total / max(1, crops_count),
            "crops": crops_count
        }
        
        print(f"Saved {crops_count} crops from {img_name} - Time: {processing_time:.3f}s")
    
    # Calculate time statistics
    timing_stats = {}
    if processing_times:
        # Overall processing stats
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        print("\n=== Cropping Time Statistics ===")
        print(f"Average processing time per image: {avg_time:.3f}s")
        print(f"Maximum processing time per image: {max_time:.3f}s")
        print(f"Minimum processing time per image: {min_time:.3f}s")
        
        # Calculate per-operation stats
        if reading_times:
            avg_reading = sum(reading_times) / len(reading_times)
            max_reading = max(reading_times)
            min_reading = min(reading_times)
            
            print("\n--- Image Reading Time ---")
            print(f"Average reading time: {avg_reading:.3f}s")
            print(f"Maximum reading time: {max_reading:.3f}s")
            print(f"Minimum reading time: {min_reading:.3f}s")
        
        if detection_times:
            avg_detection = sum(detection_times) / len(detection_times)
            max_detection = max(detection_times)
            min_detection = min(detection_times)
            
            print("\n--- Detection Time ---")
            print(f"Average detection time: {avg_detection:.3f}s")
            print(f"Maximum detection time: {max_detection:.3f}s")
            print(f"Minimum detection time: {min_detection:.3f}s")
        
        if cropping_times:
            avg_cropping = sum(cropping_times) / len(cropping_times)
            max_cropping = max(cropping_times)
            min_cropping = min(cropping_times)
            
            print("\n--- Cropping Time (per crop) ---")
            print(f"Average cropping time: {avg_cropping:.3f}s")
            print(f"Maximum cropping time: {max_cropping:.3f}s")
            print(f"Minimum cropping time: {min_cropping:.3f}s")
        
        if saving_times:
            avg_saving = sum(saving_times) / len(saving_times)
            max_saving = max(saving_times)
            min_saving = min(saving_times)
            
            print("\n--- Saving Time (per crop) ---")
            print(f"Average saving time: {avg_saving:.3f}s")
            print(f"Maximum saving time: {max_saving:.3f}s")
            print(f"Minimum saving time: {min_saving:.3f}s")
        
        # Average crops per image
        avg_crops = target_count / len(image_files)
        print(f"\nAverage crops per image: {avg_crops:.2f}")
        
        # Create timing stats dictionary
        timing_stats = {
            "avg": avg_time,
            "max": max_time,
            "min": min_time,
            "model_load": model_load_time,
            "reading": {
                "avg": avg_reading if reading_times else 0,
                "max": max_reading if reading_times else 0,
                "min": min_reading if reading_times else 0
            },
            "detection": {
                "avg": avg_detection if detection_times else 0,
                "max": max_detection if detection_times else 0,
                "min": min_detection if detection_times else 0
            },
            "cropping": {
                "avg": avg_cropping if cropping_times else 0,
                "max": max_cropping if cropping_times else 0,
                "min": min_cropping if cropping_times else 0
            },
            "saving": {
                "avg": avg_saving if saving_times else 0,
                "max": max_saving if saving_times else 0,
                "min": min_saving if saving_times else 0
            },
            "crops_per_image": crops_per_image,
            "avg_crops_per_image": avg_crops,
            "detailed": detailed_timings
        }
    
    print(f"Cropping complete. Found and saved {target_count} '{target_class}' objects.")
    return cropped_images, timing_stats

# === Process Cropped Images Function (matches barcodes from cropped images) ===
def process_cropped_images(cropped_folder, output_csv, barcode_data_csv, debug_dir=None):
    """
    Process cropped images and match with barcode_data.csv by barcode only:
    1. Read all images in cropped folder
    2. Extract barcode and three-segment code from each
    3. Match with entries in barcode_data.csv by barcode value
    4. Write results to CSV
    
    Args:
        cropped_folder: Folder containing cropped images
        output_csv: Path to save results
        barcode_data_csv: CSV with expected barcode and code values
        debug_dir: Optional directory for saving debug images
    """
    # Create debug directory if needed
    if debug_dir:
        barcode_debug_dir = os.path.join(debug_dir, "barcode_extraction")
        create_directory(barcode_debug_dir)
    
    # Read barcode data from CSV (indexed by barcode/tracking number)
    expected_data = {}
    try:
        with open(barcode_data_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                barcode = normalize(row['Tracking Number'])
                expected_data[barcode] = {
                    'three_code': normalize(row['Three-Code'])
                }
        print(f"Loaded {len(expected_data)} barcodes from {barcode_data_csv}")
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return
    
    # Get all cropped images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        image_files.extend(list(Path(cropped_folder).glob(f"*{ext}")))
        image_files.extend(list(Path(cropped_folder).glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {cropped_folder}")
        return
    
    print(f"Found {len(image_files)} cropped images to process")
    
    # Create results CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Image",
            "Detected Barcode",
            "Matched Barcode",
            "Expected Three-Code",
            "Detected Three-Code",
            "Barcode Match",
            "Three-Code Match",
            "Processing Time (s)"
        ])
        
        # Track statistics
        total_count = len(image_files)
        barcode_match_count = 0
        code_match_count = 0
        overall_success_count = 0
        processing_times = []
        
        # Process each image
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            print(f"Processing: {img_name}")
            
            start_time = time.time()
            
            try:
                # Read the image
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"Could not read image: {img_path}")
                
                # Create debug dir for this image
                img_debug_dir = None
                if barcode_debug_dir:
                    img_debug_dir = os.path.join(barcode_debug_dir, os.path.splitext(img_name)[0])
                    create_directory(img_debug_dir)
                
                # Split image into barcode and three-segment code regions
                barcode_regions, three_code_regions = split_image(img)
                
                # Process barcode regions
                detected_barcode = ""
                barcode_sim = 0
                
                for region in barcode_regions:
                    x1, y1, x2, y2 = map(int, region)
                    
                    # Ensure coordinates are within image bounds
                    height, width = img.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Extract region
                    barcode_img = img[y1:y2, x1:x2]
                    
                    # Save debug image if needed
                    if img_debug_dir:
                        cv2.imwrite(os.path.join(img_debug_dir, "barcode_region.jpg"), barcode_img)
                        enhanced = enhance_image(barcode_img, is_barcode=True)
                        cv2.imwrite(os.path.join(img_debug_dir, "enhanced_barcode.jpg"), enhanced)
                    
                    # Try to recognize barcode
                    barcode, _ = recognize_barcode(barcode_img)
                    
                    if barcode and barcode.startswith("TH"):
                        detected_barcode = barcode
                        break
                
                # Try to match detected barcode with expected barcodes
                matched_barcode = ""
                expected_code = ""
                
                if detected_barcode:
                    # Try exact match first
                    if detected_barcode in expected_data:
                        matched_barcode = detected_barcode
                        expected_code = expected_data[detected_barcode]['three_code']
                    else:
                        # Try fuzzy matching
                        best_match = ""
                        best_sim = 0
                        for exp_barcode, data in expected_data.items():
                            sim, _ = fuzzy_match(detected_barcode, exp_barcode, 0.8)
                            if sim > best_sim:
                                best_sim = sim
                                best_match = exp_barcode
                        
                        # If good enough match found
                        if best_sim >= 0.85:
                            matched_barcode = best_match
                            expected_code = expected_data[best_match]['three_code']
                            barcode_sim = best_sim
                
                # Process three-segment code regions
                detected_code = ""
                code_sim = 0
                
                for region in three_code_regions:
                    x1, y1, x2, y2 = map(int, region)
                    
                    # Ensure coordinates are within image bounds
                    height, width = img.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Extract region
                    code_img = img[y1:y2, x1:x2]
                    
                    # Save debug image if needed
                    if img_debug_dir:
                        cv2.imwrite(os.path.join(img_debug_dir, "code_region.jpg"), code_img)
                        enhanced = enhance_image(code_img, is_barcode=False)
                        cv2.imwrite(os.path.join(img_debug_dir, "enhanced_code.jpg"), enhanced)
                    
                    # Try to recognize three-segment code with expected code
                    code, sim = recognize_three_segment_code(code_img, expected_code)
                    
                    if code:
                        detected_code = code
                        code_sim = sim
                        break
                
                # Determine match status
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Determine if barcode matched
                barcode_match = "❌"
                if matched_barcode:
                    barcode_match = "✅"
                    barcode_match_count += 1
                
                # Determine if three-segment code matched
                code_match = "❌"
                if matched_barcode and detected_code:
                    # Compare with expected code
                    if expected_code and code_sim >= 0.85:
                        code_match = "✅"
                        code_match_count += 1
                
                # Count overall success
                if barcode_match == "✅" and code_match == "✅":
                    overall_success_count += 1
                
                # Write results to CSV
                writer.writerow([
                    img_name,
                    detected_barcode,
                    matched_barcode,
                    expected_code,
                    detected_code,
                    barcode_match,
                    code_match,
                    f"{processing_time:.3f}"
                ])
                
                # Log to console
                print(f"[{barcode_match}|{code_match}] {img_name} - Time: {processing_time:.3f}s")
                print(f"  Barcode: {detected_barcode} → Matched: {matched_barcode}")
                if expected_code:
                    print(f"  Three-Code: {detected_code} → Expected: {expected_code}")
                
            except Exception as e:
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                print(f"Error processing {img_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                writer.writerow([
                    img_name,
                    detected_barcode if 'detected_barcode' in locals() else "⚠️ Error",
                    matched_barcode if 'matched_barcode' in locals() else "",
                    expected_code if 'expected_code' in locals() else "",
                    detected_code if 'detected_code' in locals() else "⚠️ Error",
                    "❌",
                    "❌",
                    f"{processing_time:.2f}"
                ])
        
        # Calculate success rates
        barcode_rate = (barcode_match_count / total_count) * 100 if total_count > 0 else 0
        code_rate = (code_match_count / barcode_match_count) * 100 if barcode_match_count > 0 else 0
        overall_rate = (overall_success_count / total_count) * 100 if total_count > 0 else 0
        
        # Calculate time statistics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            min_time = min(processing_times)
        else:
            avg_time = max_time = min_time = 0
        
        print("\n=== Results ===")
        print(f"Total images processed: {total_count}")
        print(f"Barcode matches: {barcode_match_count}/{total_count} ({barcode_rate:.1f}%)")
        print(f"Three-segment code matches: {code_match_count}/{barcode_match_count} ({code_rate:.1f}%)")
        print(f"Overall success: {overall_success_count}/{total_count} ({overall_rate:.1f}%)")
        
        print("\n=== Processing Time Statistics ===")
        print(f"Average processing time: {avg_time:.3f}s")
        print(f"Maximum processing time: {max_time:.3f}s")
        print(f"Minimum processing time: {min_time:.3f}s")
        
        print(f"Results saved to: {output_csv}")

# === Full Pipeline ===
def full_pipeline(input_folder, output_folder, model_path, barcode_data_csv, results_csv, target_class="Target", debug_dir=None):
    """
    Run full processing pipeline:
    1. Detect objects using YOLO
    2. Crop detected targets
    3. Process cropped images to extract barcodes and match with expected values
    
    Args:
        input_folder: Folder with original images
        output_folder: Base folder for outputs
        model_path: Path to YOLO model
        barcode_data_csv: CSV with expected barcode and code values
        results_csv: Path to save results
        target_class: Class to detect
        debug_dir: Directory for debug images
    """
    # Create output directories
    create_directory(output_folder)
    
    # Create subdirectories
    detected_folder = os.path.join(output_folder, "detected")
    cropped_folder = os.path.join(output_folder, "cropped")
    
    # Create debug directory if needed
    if debug_dir:
        create_directory(debug_dir)
    
    # Dictionary to store all timing statistics
    pipeline_stats = {}
    
    print("\n=== Step 1: YOLO Detection ===")
    yolo_start = time.time()
    # Detect objects and save images with bounding boxes
    detected_images, yolo_stats = detect_and_save_images(
        input_folder, 
        detected_folder, 
        model_path,
        debug_dir
    )
    yolo_total = time.time() - yolo_start
    pipeline_stats["yolo"] = yolo_stats
    pipeline_stats["yolo"]["total_time"] = yolo_total
    
    print("\n=== Step 2: Cropping Detected Targets ===")
    crop_start = time.time()
    # Crop targets from detected images
    cropped_images, crop_stats = crop_objects(
        input_folder,  # Use original images for better quality
        cropped_folder, 
        model_path, 
        target_class,
        debug_dir
    )
    crop_total = time.time() - crop_start
    pipeline_stats["crop"] = crop_stats
    pipeline_stats["crop"]["total_time"] = crop_total
    
    print("\n=== Step 3: Barcode and Code Recognition ===")
    barcode_start = time.time()
    # Process the cropped labels to extract barcode and three-segment code
    # Match with entries in barcode_data.csv based on barcode value (not filename)
    process_cropped_images(
        cropped_folder,
        results_csv,
        barcode_data_csv,
        debug_dir
    )
    barcode_total = time.time() - barcode_start
    
    total_time = yolo_total + crop_total + barcode_total
    pipeline_stats["barcode"] = {"total_time": barcode_total}
    pipeline_stats["total_time"] = total_time
    
    # Calculate per-image processing time across the whole pipeline
    if detected_images and cropped_images:
        unique_source_images = set([os.path.splitext(os.path.basename(path))[0].split('_')[0] 
                                   for path in detected_images + cropped_images])
        avg_time_per_image = total_time / len(unique_source_images) if unique_source_images else 0
        pipeline_stats["avg_time_per_image"] = avg_time_per_image
    
    print("\n=== Pipeline Complete ===")
    print(f"📊 Detected images saved to: {detected_folder}")
    print(f"📊 Cropped targets saved to: {cropped_folder}")
    print(f"📊 Results saved to: {results_csv}")
    if debug_dir:
        print(f"📊 Debug images saved to: {debug_dir}")
    
    print("\n=== Overall Processing Time ===")
    print(f"YOLO Detection: {yolo_total:.2f}s")
    print(f"Target Cropping: {crop_total:.2f}s")
    print(f"Barcode Processing: {barcode_total:.2f}s")
    print(f"Total Pipeline Time: {total_time:.2f}s")
    
    # If we have processed images, calculate the average time per image
    if "avg_time_per_image" in pipeline_stats:
        print(f"Average time per source image: {pipeline_stats['avg_time_per_image']:.2f}s")
    
    # Write timing statistics to CSV
    try:
        stats_csv = os.path.join(output_folder, "timing_statistics.csv")
        with open(stats_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Process", "Average Time (s)", "Maximum Time (s)", "Minimum Time (s)"])
            
            # YOLO detection stats
            writer.writerow(["YOLO Detection (total)", 
                            f"{yolo_stats.get('avg', 0):.3f}", 
                            f"{yolo_stats.get('max', 0):.3f}", 
                            f"{yolo_stats.get('min', 0):.3f}"])
            
            if "detection" in yolo_stats:
                writer.writerow(["YOLO Inference", 
                                f"{yolo_stats['detection'].get('avg', 0):.3f}", 
                                f"{yolo_stats['detection'].get('max', 0):.3f}", 
                                f"{yolo_stats['detection'].get('min', 0):.3f}"])
            
            if "saving" in yolo_stats:
                writer.writerow(["YOLO Result Saving", 
                                f"{yolo_stats['saving'].get('avg', 0):.3f}", 
                                f"{yolo_stats['saving'].get('max', 0):.3f}", 
                                f"{yolo_stats['saving'].get('min', 0):.3f}"])
            
            # Cropping stats
            writer.writerow(["Cropping (total)", 
                            f"{crop_stats.get('avg', 0):.3f}", 
                            f"{crop_stats.get('max', 0):.3f}", 
                            f"{crop_stats.get('min', 0):.3f}"])
            
            if "reading" in crop_stats:
                writer.writerow(["Image Reading", 
                                f"{crop_stats['reading'].get('avg', 0):.3f}", 
                                f"{crop_stats['reading'].get('max', 0):.3f}", 
                                f"{crop_stats['reading'].get('min', 0):.3f}"])
            
            if "detection" in crop_stats:
                writer.writerow(["Target Detection", 
                                f"{crop_stats['detection'].get('avg', 0):.3f}", 
                                f"{crop_stats['detection'].get('max', 0):.3f}", 
                                f"{crop_stats['detection'].get('min', 0):.3f}"])
            
            if "cropping" in crop_stats:
                writer.writerow(["Image Cropping", 
                                f"{crop_stats['cropping'].get('avg', 0):.3f}", 
                                f"{crop_stats['cropping'].get('max', 0):.3f}", 
                                f"{crop_stats['cropping'].get('min', 0):.3f}"])
            
            if "saving" in crop_stats:
                writer.writerow(["Crop Saving", 
                                f"{crop_stats['saving'].get('avg', 0):.3f}", 
                                f"{crop_stats['saving'].get('max', 0):.3f}", 
                                f"{crop_stats['saving'].get('min', 0):.3f}"])
            
            # Overall pipeline stats
            writer.writerow(["Total Pipeline", f"{total_time:.3f}", "", ""])
        
        print(f"📊 Timing statistics saved to: {stats_csv}")
    except Exception as e:
        print(f"Error saving timing statistics: {str(e)}")
        
    return pipeline_stats

# === Main ===
if __name__ == "__main__":
    # Configuration
    input_folder = "label"  # Folder with original images
    output_folder = "output"  # Base folder for outputs
    model_path = "best.pt"  # YOLO model path
    barcode_data_csv = "barcode_data.csv"  # CSV with expected values
    results_csv = "barcode_result.csv"  # Results CSV
    debug_dir = "ocr_debug"  # Debug images directory
    
    # Start timing the entire process
    overall_start_time = time.time()
    
    # Run the complete pipeline
    pipeline_stats = full_pipeline(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=model_path,
        barcode_data_csv=barcode_data_csv,
        results_csv=results_csv,
        target_class="Target",
        debug_dir=debug_dir
    )
    
    # Calculate total execution time
    overall_execution_time = time.time() - overall_start_time
    
    print(f"\n=== Script Complete ===")
    print(f"Total execution time: {overall_execution_time:.3f}s")
    
    # Create a visual summary with timing breakdown
    print("\n=== Time Breakdown Summary ===")
    yolo_time = pipeline_stats.get("yolo", {}).get("total_time", 0)
    crop_time = pipeline_stats.get("crop", {}).get("total_time", 0)
    barcode_time = pipeline_stats.get("barcode", {}).get("total_time", 0)
    
    # Calculate percentages
    total_pipeline_time = yolo_time + crop_time + barcode_time
    yolo_percent = (yolo_time / total_pipeline_time) * 100 if total_pipeline_time > 0 else 0
    crop_percent = (crop_time / total_pipeline_time) * 100 if total_pipeline_time > 0 else 0
    barcode_percent = (barcode_time / total_pipeline_time) * 100 if total_pipeline_time > 0 else 0
    
    # Print simple bar chart
    print(f"YOLO Detection  : {yolo_time:.3f}s ({yolo_percent:.1f}%) {'█' * int(yolo_percent / 2)}")
    print(f"Target Cropping : {crop_time:.3f}s ({crop_percent:.1f}%) {'█' * int(crop_percent / 2)}")
    print(f"Barcode Process : {barcode_time:.3f}s ({barcode_percent:.1f}%) {'█' * int(barcode_percent / 2)}")
    print(f"Total Pipeline  : {total_pipeline_time:.3f}s")
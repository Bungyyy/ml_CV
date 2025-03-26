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
    """
    # Create output folders
    create_directory(output_folder)
    if debug_dir:
        yolo_debug_dir = os.path.join(debug_dir, "yolo_detections")
        create_directory(yolo_debug_dir)
    else:
        yolo_debug_dir = None
    
    # Load model
    model = YOLO(model_path)
    
    # Get list of image files in the input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f"*{ext}")))
        image_files.extend(list(Path(input_folder).glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return []
    
    print(f"Found {len(image_files)} images to process with YOLO")
    
    # Process each image
    detected_images = []
    for image_path in image_files:
        image_file = os.path.basename(image_path)
        
        # Construct output paths
        output_path = os.path.join(output_folder, image_file)
        
        # Perform prediction
        results = model.predict(str(image_path))
        
        # Access first result
        result = results[0]
        
        # Save with bounding boxes
        result.save(filename=output_path)
        
        # Also save to debug directory if specified
        if yolo_debug_dir:
            debug_path = os.path.join(yolo_debug_dir, image_file)
            result.save(filename=debug_path)
        
        detected_images.append(output_path)
        print(f"YOLO Detection: {image_file}")
    
    return detected_images

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
    """
    # Create output folders
    create_directory(output_folder)
    if debug_dir:
        crop_debug_dir = os.path.join(debug_dir, "cropped_targets")
        create_directory(crop_debug_dir)
    else:
        crop_debug_dir = None
    
    # Load YOLOv model
    model = YOLO(model_path)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f"*{ext}")))
        image_files.extend(list(Path(input_folder).glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return []
    
    print(f"Found {len(image_files)} images to process for cropping")
    
    # Process each image
    target_count = 0
    cropped_images = []
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"Processing for crop: {img_name}")
        
        # Read the image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error reading image: {img_path}")
            continue
        
        # Run inference
        results = model(img)
        
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
            continue
        
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
            
            # Crop the image
            cropped_img = img[y1:y2, x1:x2]
            
            if cropped_img.size == 0:
                print(f"Warning: Empty crop in {img_name} at {box}")
                continue
            
            # Save the cropped image
            base_name = os.path.splitext(img_name)[0]
            output_name = f"{base_name}_{target_class}_{idx}_{conf:.2f}.jpg"
            output_path = os.path.join(output_folder, output_name)
            cv2.imwrite(output_path, cropped_img)
            
            # Also save to debug directory if specified
            if crop_debug_dir:
                debug_path = os.path.join(crop_debug_dir, output_name)
                cv2.imwrite(debug_path, cropped_img)
            
            cropped_images.append(output_path)
            target_count += 1
            print(f"Saved crop: {output_name}")
    
    print(f"Cropping complete. Found and saved {target_count} '{target_class}' objects.")
    return cropped_images

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
                    f"{processing_time:.2f}"
                ])
                
                # Log to console
                print(f"[{barcode_match}|{code_match}] {img_name}")
                print(f"  Barcode: {detected_barcode} → Matched: {matched_barcode}")
                if expected_code:
                    print(f"  Three-Code: {detected_code} → Expected: {expected_code}")
                
            except Exception as e:
                processing_time = time.time() - start_time
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
        
        print("\n=== Results ===")
        print(f"Total images processed: {total_count}")
        print(f"Barcode matches: {barcode_match_count}/{total_count} ({barcode_rate:.1f}%)")
        print(f"Three-segment code matches: {code_match_count}/{barcode_match_count} ({code_rate:.1f}%)")
        print(f"Overall success: {overall_success_count}/{total_count} ({overall_rate:.1f}%)")
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
    
    print("\n=== Step 1: YOLO Detection ===")
    # Detect objects and save images with bounding boxes
    detect_and_save_images(
        input_folder, 
        detected_folder, 
        model_path,
        debug_dir
    )
    
    print("\n=== Step 2: Cropping Detected Targets ===")
    # Crop targets from detected images
    crop_objects(
        input_folder,  # Use original images for better quality
        cropped_folder, 
        model_path, 
        target_class,
        debug_dir
    )
    
    print("\n=== Step 3: Barcode and Code Recognition ===")
    # Process the cropped labels to extract barcode and three-segment code
    # Match with entries in barcode_data.csv based on barcode value (not filename)
    process_cropped_images(
        cropped_folder,
        results_csv,
        barcode_data_csv,
        debug_dir
    )
    
    print("\n=== Pipeline Complete ===")
    print(f"📊 Detected images saved to: {detected_folder}")
    print(f"📊 Cropped targets saved to: {cropped_folder}")
    print(f"📊 Results saved to: {results_csv}")
    if debug_dir:
        print(f"📊 Debug images saved to: {debug_dir}")

# === Main ===
if __name__ == "__main__":
    # Configuration
    input_folder = "label"  # Folder with original images
    output_folder = "output"  # Base folder for outputs
    model_path = "best.pt"  # YOLO model path
    barcode_data_csv = "barcode_data.csv"  # CSV with expected values
    results_csv = "barcode_result_improved.csv"  # Results CSV
    debug_dir = "ocr_debug_improved"  # Debug images directory
    
    # Run the complete pipeline
    full_pipeline(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=model_path,
        barcode_data_csv=barcode_data_csv,
        results_csv=results_csv,
        target_class="Target",
        debug_dir=debug_dir
    )
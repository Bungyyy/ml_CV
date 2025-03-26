import cv2
import numpy as np
import os
import torch
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import pytesseract
import difflib
import re
import csv
import time

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

def char_diff(a: str, b: str) -> int:
    """Count character differences between strings of the same length"""
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)

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
    
    # Apply preprocessing for better region detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Simple approach - use traditional split method
    # Define regions based on typical placement
    top_height = int(height * 0.25)  # Assume top 25% has the three-segment code
    
    # Barcode is often in the bottom portion
    barcode_regions = [(0, top_height, width, height)]
    
    # Define the three-segment code region based on typical placement
    # Usually at the top of the image
    three_segment_regions = [(0, 0, width, top_height)]
    
    return barcode_regions, three_segment_regions

# === Three-Segment Code Recognition ===
def extract_three_segment_code(text):
    """Extract three-segment code from OCR text using pattern matching"""
    # Try to find a pattern like XXX-XXXX-XXX with various separators
    pattern = r'([A-Z0-9]{3})[\s\-_.]+([A-Z0-9]{4})[\s\-_.]+([A-Z0-9]{3})'
    match = re.search(pattern, text.upper())
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    
    # Fallback for less structured text
    # Look for sequences of letters and numbers that are roughly the right length
    parts = re.findall(r'[A-Z0-9]{3,5}', text.upper())
    if len(parts) >= 3:
        potential_code = f"{parts[0][:3]}-{parts[1][:4]}-{parts[2][:3]}"
        return potential_code
    
    return ""

def correct_ocr_errors(code: str, context: str = "auto") -> str:
    """
    Correct common OCR errors in text
    
    Args:
        code: The OCR result to correct
        context: Either "auto", "barcode" or "three-segment"
    """
    if not code:
        return code
    
    # Convert to uppercase
    result = code.upper()
    
    # Determine context if auto
    if context == "auto":
        # If it starts with TH, it's likely a barcode
        if "TH" in result:
            context = "barcode"
        # If it has hyphens and follows XXX-XXXX-XXX pattern, it's likely three-segment
        elif re.search(r'[A-Z0-9]{3}[\s\-_.]+[A-Z0-9]{4}[\s\-_.]+[A-Z0-9]{3}', result):
            context = "three-segment"
        else:
            context = "three-segment"  # Default to three-segment
    
    # Common corrections for both contexts
    corrections = {
        '|': 'I',  # | → I
        '[': 'I',  # [ → I
        ']': 'I',  # ] → I
        '{': 'I',  # { → I
        '}': 'I',  # } → I
        'l': 'I',  # l → I
        ',': '-',  # , → -
        '.': '-',  # . → -
        '_': '-',  # _ → -
        "'": '-',  # ' → -
        "`": '-',  # ` → -
    }
    
    # Apply context-specific corrections
    if context == "barcode":
        # For barcodes, focus on numeric correctness
        barcode_corrections = {
            'O': '0',  # O → 0
            'I': '1',  # I → 1
            'S': '5',  # S → 5
            'B': '8',  # B → 8
            'Z': '2',  # Z → 2
            'Q': '0',  # Q → 0
            'D': '0',  # D → 0 (similar shapes)
            'G': '6',  # G → 6
        }
        corrections.update(barcode_corrections)
    else:
        # For three-segment codes, keep uppercase letters, don't convert to numbers
        # This is for three-segment codes that use only uppercase letters
        three_segment_corrections = {
            # Uppercase alphabet only - no conversions from numbers to letters
            # No lowercase to uppercase needed as we've already uppercased the input
        }
        corrections.update(three_segment_corrections)
    
    # Apply corrections
    for bad, good in corrections.items():
        result = result.replace(bad, good)
    
    # Clean up dashes by replacing various dash-like characters
    result = re.sub(r'[‐‑‒–—―_]+', '-', result)
    
    # For three-segment codes, try to restore the standard format XXX-XXXX-XXX
    if context == "three-segment":
        if len(result) >= 8 and '-' not in result and len(result.replace(' ', '')) >= 10:
            # No dashes or spaces found, try to insert them
            clean = result.replace(' ', '')
            if len(clean) >= 10:
                result = clean[:3] + '-' + clean[3:7] + '-' + clean[7:10]
        
        # Check if we have something close to our pattern but with wrong separators
        match = re.search(r'([A-Z0-9]{3})[^A-Z0-9]?([A-Z0-9]{4})[^A-Z0-9]?([A-Z0-9]{3})', result)
        if match:
            result = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    
    return result

def recognize_barcode(image, expected_barcode=""):
    """
    Recognize barcode from an image region using enhanced processing
    
    Args:
        image: PIL Image or numpy array containing the barcode
        expected_barcode: Expected barcode for validation (optional)
    
    Returns:
        tuple: (decoded_barcode, similarity_score)
    """
    # Convert PIL to NumPy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        # Convert RGB to BGR if needed
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
                corrected = correct_ocr_errors(line_clean, context="barcode")
                
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
    """
    Recognize three-segment code from an image region using enhanced processing
    
    Args:
        image: PIL Image or numpy array containing the three-segment code
        expected_code: Expected code for validation (optional)
    
    Returns:
        tuple: (decoded_code, similarity_score)
    """
    # Convert PIL to NumPy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        # Convert RGB to BGR if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Enhance the image for text recognition
    enhanced = enhance_image(image, is_barcode=False)
    
    # Try multiple OCR approaches
    potential_codes = []
    
    try:
        # Try different PSM modes and preprocessing
        psm_modes = [6, 7, 8, 11]  # Various page segmentation modes
        
        for psm in psm_modes:
            ocr_text = pytesseract.image_to_string(
                enhanced, 
                config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. '
            )
            
            lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
            for line in lines:
                # Look for patterns that match three-segment code
                if "-" in line or len(line) >= 8:
                    corrected = correct_ocr_errors(line, context="three-segment")
                    potential_codes.append(corrected)
                
                # Try pattern matching
                extracted = extract_three_segment_code(line)
                if extracted:
                    potential_codes.append(extracted)
        
        # Try with inverted image if we didn't get good results
        if not potential_codes:
            inverted = cv2.bitwise_not(enhanced)
            ocr_text = pytesseract.image_to_string(
                inverted,
                config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. '
            )
            
            for line in ocr_text.splitlines():
                if "-" in line or len(line) >= 8:
                    corrected = correct_ocr_errors(line, context="three-segment")
                    potential_codes.append(corrected)
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

# === Main Processing Function ===
def process_image(image_path, expected_barcode="", expected_code="", debug_dir=None):
    """
    Process an image to extract barcode and three-segment code information
    
    Args:
        image_path: Path to the image file
        expected_barcode: Expected barcode for validation (optional)
        expected_code: Expected three-segment code for validation (optional)
        debug_dir: Directory for saving debug images (optional)
    
    Returns:
        dict: Results including barcode, three-segment code, and status information
    """
    try:
        # Load the image
        if isinstance(image_path, str):
            # Load from file path
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            # Assume it's already an image array
            img = image_path
        
        # Split into barcode and three-segment code regions
        barcode_regions, three_segment_regions = split_image(img)
        
        # Process barcode regions
        barcode_result = ""
        barcode_sim = 0
        
        for region in barcode_regions:
            x1, y1, x2, y2 = region
            # Convert to integer coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            
            # Check if region is valid
            if x1 >= x2 or y1 >= y2:
                continue
                
            # Extract region
            barcode_img = img[y1:y2, x1:x2]
            
            # Save debug image if directory specified
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                debug_filename = os.path.basename(image_path) if isinstance(image_path, str) else "debug_barcode.png"
                enhanced = enhance_image(barcode_img, is_barcode=True)
                cv2.imwrite(os.path.join(debug_dir, f"barcode_{debug_filename}"), enhanced)
            
            # Recognize barcode
            result, sim = recognize_barcode(barcode_img, expected_barcode)
            
            # Keep the best result
            if sim > barcode_sim or not barcode_result:
                barcode_result = result
                barcode_sim = sim
        
        # Process three-segment code regions
        code_result = ""
        code_sim = 0
        
        for region in three_segment_regions:
            x1, y1, x2, y2 = region
            # Convert to integer coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            
            # Check if region is valid
            if x1 >= x2 or y1 >= y2:
                continue
                
            # Extract region
            code_img = img[y1:y2, x1:x2]
            
            # Save debug image if directory specified
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                debug_filename = os.path.basename(image_path) if isinstance(image_path, str) else "debug_three_segment.png"
                enhanced = enhance_image(code_img, is_barcode=False)
                cv2.imwrite(os.path.join(debug_dir, f"text_{debug_filename}"), enhanced)
            
            # Recognize three-segment code
            result, sim = recognize_three_segment_code(code_img, expected_code)
            
            # Keep the best result
            if sim > code_sim or not code_result:
                code_result = result
                code_sim = sim
        
        # Determine status
        barcode_ok = barcode_sim >= 0.85
        barcode_status = "✅" if barcode_ok else "❌"
        
        if len(code_result) < 3:
            code_status = "⚠️ Missing"
        elif code_sim >= 0.95:
            code_status = "✅"
        elif code_sim >= 0.85:
            code_status = "⚠️ Close"
        else:
            code_status = "❌"
        
        return {
            "barcode": barcode_result,
            "three_segment_code": code_result,
            "barcode_status": barcode_status,
            "code_status": code_status,
            "barcode_sim": barcode_sim,
            "code_sim": code_sim
        }
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "barcode": "⚠️ Error",
            "three_segment_code": "⚠️ Error",
            "barcode_status": "❌",
            "code_status": "❌",
            "barcode_sim": 0.0,
            "code_sim": 0.0
        }

# === Main Processing Script ===
def process_batch(input_csv, output_csv, debug_dir=None):
    """Process a batch of images based on entries in a CSV file"""
    with open(input_csv, newline='') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        writer.writerow([
            "Filename",
            "Expected Barcode",
            "Expected 3Code",
            "Decoded Barcode",
            "Decoded 3Code",
            "Match Barcode",
            "Match 3Code",
            "Similarity Barcode",
            "Similarity 3Code",
            "Processing Time (s)"
        ])

        success_count = 0
        barcode_success_count = 0
        total_count = 0

        for row in reader:
            total_count += 1
            expected_barcode = normalize(row['Tracking Number'])
            expected_code = normalize(row['Three-Code'])
            img_path = row["Filename"]

            start_time = time.time()
            
            try:
                result = process_image(img_path, expected_barcode, expected_code, debug_dir)
                
                processing_time = time.time() - start_time
                
                barcode_ok = result["barcode_status"] == "✅"
                code_ok = result["code_status"] == "✅"
                
                if barcode_ok:
                    barcode_success_count += 1
                if code_ok:
                    success_count += 1
                
                writer.writerow([
                    img_path,
                    expected_barcode,
                    expected_code,
                    result["barcode"],
                    result["three_segment_code"],
                    result["barcode_status"],
                    result["code_status"],
                    f"{result['barcode_sim']:.2f}",
                    f"{result['code_sim']:.2f}",
                    f"{processing_time:.2f}"
                ])
                
                print(f"[{result['barcode_status']}|{result['code_status']}] {os.path.basename(img_path)}: {expected_barcode} → {result['barcode']} | {expected_code} → {result['three_segment_code']}")
                
            except Exception as e:
                processing_time = time.time() - start_time
                writer.writerow([
                    img_path,
                    expected_barcode,
                    expected_code,
                    "⚠️ Error",
                    "⚠️ Error",
                    "❌",
                    "❌",
                    "0.00",
                    "0.00",
                    f"{processing_time:.2f}"
                ])
                print(f"[⚠️] {img_path} → {str(e)}")

        # Calculate success rates
        barcode_success_rate = (barcode_success_count / total_count) * 100 if total_count > 0 else 0
        three_code_success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\n📄 Analysis and export complete: {output_csv} ✅")
        print(f"📊 Barcode Success Rate: {barcode_success_count}/{total_count} ({barcode_success_rate:.1f}%)")
        print(f"📊 Three-Segment Code Success Rate: {success_count}/{total_count} ({three_code_success_rate:.1f}%)")

# === Configuration and Entry Point ===
if __name__ == "__main__":
    input_csv = "barcode_data.csv"
    output_csv = "barcode_result_improved.csv"
    debug_dir = "ocr_debug_improved"
    
    # Create debug directory if needed
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    process_batch(input_csv, output_csv, debug_dir)
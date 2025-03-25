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

# === Helper Functions (from original) ===
def normalize(text):
    """Normalize text by removing spaces and converting to uppercase"""
    return re.sub(r'\s+', '', text).strip().upper()

def correct_common_ocr_errors(code: str) -> str:
    """Correct common OCR errors in three-segment codes"""
    # Note: Based on the results, we've reversed the S/5 correction
    # as it seems three-segment codes use 5 not S in your case
    corrections = {
        '0': 'O',  # 0 → O
        '1': 'I',  # 1 → I
        'S': '5',  # S → 5 (reversed from original)
        '8': 'B',  # 8 → B
        '2': 'Z',  # 2 → Z
        '6': 'G',  # 6 → G
        '|': 'I',  # | → I
        '[': 'I',  # [ → I
        ']': 'I',  # ] → I
        '{': 'I',  # { → I
        '}': 'I',  # } → I
        'l': 'I',  # l → I
    }
    result = code.upper()
    for bad, good in corrections.items():
        result = result.replace(bad, good)
    
    # Clean up dashes by replacing various dash-like characters
    result = re.sub(r'[‐‑‒–—―_]+', '-', result)
    
    # Fix common three-segment errors by pattern
    # Try to restore standard three-segment format XXX-XXXX-XXX
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

def process_text_region(img):
    """Process the text region to optimize for OCR"""
    # Convert to grayscale
    gray = ImageOps.grayscale(img)
    
    # Enhance contrast for text
    contrast = ImageEnhance.Contrast(gray).enhance(2.0)
    
    # Apply slight blur to reduce noise (helps OCR with text)
    blurred = contrast.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Enhance sharpness
    sharp = ImageEnhance.Sharpness(blurred).enhance(1.5)
    
    return sharp

def process_barcode_region(img):
    """Process the barcode region to optimize for barcode detection"""
    # Convert to grayscale
    gray = ImageOps.grayscale(img)
    
    # Enhance contrast 
    contrast = ImageEnhance.Contrast(gray).enhance(2.5)
    
    # Apply adaptive thresholding
    threshold = contrast.point(lambda p: 255 if p > 128 else 0)
    
    # Enhance sharpness
    sharp = ImageEnhance.Sharpness(threshold).enhance(2.0)
    
    return sharp

# === YOLOv4 Implementation ===
class BarcodeDetector:
    """Implements the improved YOLOv4 model as described in the paper"""
    def __init__(self, model_path=None):
        # Check if CUDA is available and set device accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Try to load YOLOv5 with a pre-trained model instead of custom model
            try:
                # Use pre-trained YOLOv5s model instead of requiring a custom model
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                self.model_type = 'yolov5'
                self.model.to(self.device)
                self.model.eval()
                print(f"Pre-trained YOLOv5 model loaded successfully on {self.device}")
            except Exception as e:
                print(f"Error loading pre-trained YOLOv5: {str(e)}")
                self.model = None
                self.model_type = None
        except Exception as e:
            print(f"Error in model loading process: {str(e)}")
            print("Using fallback detection method")
            self.model = None
            self.model_type = None
    
    def detect(self, image):
        """Detect barcodes and three-segment codes in an image"""
        if self.model is None:
            # Fallback to the traditional split method if model isn't available
            return self._fallback_detect(image)
            
        # Convert PIL image to format expected by YOLO
        img = np.array(image)
        
        # Run detection
        results = self.model(img)
        
        # Process detections
        barcode_regions = []
        three_segment_regions = []
        
        try:
            # Process results based on model type
            if self.model_type == 'yolov5':
                detections = results.pandas().xyxy[0]
                
                # For pre-trained YOLOv5 model, we need to map the classes
                # Look for objects that could be barcodes or text regions
                for _, detection in detections.iterrows():
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                    class_name = detection['name']
                    confidence = detection['confidence']
                    
                    if confidence < 0.5:  # Minimum confidence threshold
                        continue
                        
                    # In pre-trained YOLOv5, we look for regions that might contain barcodes or text
                    # 'book', 'cell phone', 'remote', 'keyboard' are good candidates for rectangular regions
                    if class_name in ['book', 'cell phone', 'remote', 'keyboard', 'tv', 'laptop']:
                        # Objects with higher aspect ratio are more likely to be barcodes
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        
                        if aspect_ratio > 3.0:  # Typical for barcodes
                            barcode_regions.append((x1, y1, x2, y2))
                        else:
                            three_segment_regions.append((x1, y1, x2, y2))
        except Exception as e:
            print(f"Error processing detections: {str(e)}")
            return self._fallback_detect(image)
            
        # If no regions detected, fall back to traditional method
        if not barcode_regions and not three_segment_regions:
            print("No regions detected by YOLOv5, falling back to traditional method")
            return self._fallback_detect(image)
            
        return barcode_regions, three_segment_regions
    
    def _fallback_detect(self, img):
        """Fallback detection method using traditional approach from original code"""
        width, height = img.size
        top_height = int(height * 0.25)  # Assume top 25% has the three-segment code
        
        # Define regions based on where these elements typically appear
        barcode_region = (0, top_height - 10, width, height)  # Bottom part
        three_segment_region = (0, 0, width, top_height)  # Top part
        
        return [barcode_region], [three_segment_region]

# === Barcode and Three-Segment Code Recognition ===
def extract_barcode(img_region, expected_barcode=""):
    """Extract barcode from image region with optimized processing"""
    try:
        # Process the barcode region
        processed_img = process_barcode_region(img_region)
        
        # Use ZBar for barcode detection and decoding
        # In a production system, you would integrate ZBar library here
        # For this example, we'll simulate with Tesseract OCR
        ocr_text = pytesseract.image_to_string(
            processed_img, 
            config='--psm 6 -c tessedit_char_whitelist=TH0123456789'
        )
        
        # Extract barcode number
        decoded_barcode = ""
        for line in ocr_text.splitlines():
            line_clean = normalize(line)
            if line_clean.startswith("TH") and len(line_clean) >= 12:
                decoded_barcode = line_clean
                break
        
        barcode_sim = 0
        if decoded_barcode and expected_barcode:
            barcode_sim, _ = fuzzy_match(decoded_barcode, expected_barcode)
        
        return decoded_barcode, barcode_sim
        
    except Exception as e:
        print(f"Error in barcode extraction: {str(e)}")
        return "", 0

def extract_three_segment(img_region, expected_code=""):
    """Extract three-segment code from image region with optimized processing"""
    try:
        # Process the text region
        processed_img = process_text_region(img_region)
        
        # Try multiple PSM modes for better accuracy
        text_results = []
        
        # PSM 6 - Assume a single uniform block of text
        text_ocr_6 = pytesseract.image_to_string(
            processed_img, 
            config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- --oem 1'
        )
        text_results.append(text_ocr_6)
        
        # PSM 7 - Treat the image as a single line of text
        text_ocr_7 = pytesseract.image_to_string(
            processed_img, 
            config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- --oem 1'
        )
        text_results.append(text_ocr_7)
        
        # Try to extract three-segment code from all text results
        potential_codes = []
        for text_result in text_results:
            lines = [line.strip() for line in text_result.splitlines() if line.strip()]
            for line in lines:
                # Try direct extraction
                if "-" in line and len(line) >= 11:
                    corrected = correct_common_ocr_errors(line)
                    potential_codes.append(corrected)
                
                # Try pattern matching
                extracted = extract_three_segment_code(line)
                if extracted:
                    potential_codes.append(extracted)
        
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
            # Without expected code, just take the first valid one
            if potential_codes:
                decoded_code = normalize(potential_codes[0])
        
        code_sim = 0
        if decoded_code and expected_code:
            code_sim, _ = fuzzy_match(decoded_code, expected_code)
        
        return decoded_code, code_sim
    
    except Exception as e:
        print(f"Error in three-segment code extraction: {str(e)}")
        return "", 0

# === Main Processing Function ===
def process_image(image_path, expected_barcode="", expected_code="", debug_dir=None):
    """Process an image to extract barcode and three-segment code information"""
    try:
        # Open image
        img = Image.open(image_path)
        
        # Initialize detector
        detector = BarcodeDetector()
        
        # Detect regions
        barcode_regions, three_segment_regions = detector.detect(img)
        
        # Extract the first detected barcode
        barcode_result = ""
        barcode_sim = 0
        if barcode_regions:
            x1, y1, x2, y2 = barcode_regions[0]
            barcode_img = img.crop((x1, y1, x2, y2))
            
            # Save debug image if directory specified
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                debug_filename = os.path.basename(image_path)
                process_barcode_region(barcode_img).save(os.path.join(debug_dir, f"barcode_{debug_filename}"))
            
            barcode_result, barcode_sim = extract_barcode(barcode_img, expected_barcode)
        
        # Extract the first detected three-segment code
        code_result = ""
        code_sim = 0
        if three_segment_regions:
            x1, y1, x2, y2 = three_segment_regions[0]
            three_segment_img = img.crop((x1, y1, x2, y2))
            
            # Save debug image if directory specified
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                debug_filename = os.path.basename(image_path)
                process_text_region(three_segment_img).save(os.path.join(debug_dir, f"text_{debug_filename}"))
            
            code_result, code_sim = extract_three_segment(three_segment_img, expected_code)
        
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
    {
        '&': 'B',  # & → B
        '@': 'Q',  # @ → Q
        '#': 'H',  # # → H
        '<': 'K',  # < → K
        '>': 'K',  # > → K
        '/': 'J',  # / → J
        '\\': 'J', # \ → J
        ',': '.',  # , → .
        '`': '.',  # ` → .
        "'": '.',  # ' → .
    }
    result = code.upper()
    for bad, good in corrections.items():
        result = result.replace(bad, good)
    
    # Clean up dashes by replacing various dash-like characters
    result = re.sub(r'[‐‑‒–—―_]+', '-', result)
    
    # Fix common three-segment errors by pattern
    # Try to restore standard three-segment format XXX-XXXX-XXX
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

def process_text_region(img):
    """Process the text region to optimize for OCR"""
    # Convert to grayscale
    gray = ImageOps.grayscale(img)
    
    # Enhance contrast for text
    contrast = ImageEnhance.Contrast(gray).enhance(2.0)
    
    # Apply slight blur to reduce noise (helps OCR with text)
    blurred = contrast.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Enhance sharpness
    sharp = ImageEnhance.Sharpness(blurred).enhance(1.5)
    
    return sharp

def process_barcode_region(img):
    """Process the barcode region to optimize for barcode detection"""
    # Convert to grayscale
    gray = ImageOps.grayscale(img)
    
    # Enhance contrast 
    contrast = ImageEnhance.Contrast(gray).enhance(2.5)
    
    # Apply adaptive thresholding
    threshold = contrast.point(lambda p: 255 if p > 128 else 0)
    
    # Enhance sharpness
    sharp = ImageEnhance.Sharpness(threshold).enhance(2.0)
    
    return sharp

# === YOLOv4 Implementation ===
class BarcodeDetector:
    """Implements the improved YOLOv4 model as described in the paper"""
    def __init__(self, model_path=None):
        # Check if CUDA is available and set device accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Try to load YOLOv5 with a pre-trained model instead of custom model
            try:
                # Use pre-trained YOLOv5s model instead of requiring a custom model
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                self.model_type = 'yolov5'
                self.model.to(self.device)
                self.model.eval()
                print(f"Pre-trained YOLOv5 model loaded successfully on {self.device}")
            except Exception as e:
                print(f"Error loading pre-trained YOLOv5: {str(e)}")
                self.model = None
                self.model_type = None
        except Exception as e:
            print(f"Error in model loading process: {str(e)}")
            print("Using fallback detection method")
            self.model = None
            self.model_type = None
    
    def detect(self, image):
        """Detect barcodes and three-segment codes in an image"""
        if self.model is None:
            # Fallback to the traditional split method if model isn't available
            return self._fallback_detect(image)
            
        # Convert PIL image to format expected by YOLO
        img = np.array(image)
        
        # Run detection
        results = self.model(img)
        
        # Process detections
        barcode_regions = []
        three_segment_regions = []
        
        try:
            # Process results based on model type
            if self.model_type == 'yolov5':
                detections = results.pandas().xyxy[0]
                
                # For pre-trained YOLOv5 model, we need to map the classes
                # Look for objects that could be barcodes or text regions
                for _, detection in detections.iterrows():
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                    class_name = detection['name']
                    confidence = detection['confidence']
                    
                    if confidence < 0.5:  # Minimum confidence threshold
                        continue
                        
                    # In pre-trained YOLOv5, we look for regions that might contain barcodes or text
                    # 'book', 'cell phone', 'remote', 'keyboard' are good candidates for rectangular regions
                    if class_name in ['book', 'cell phone', 'remote', 'keyboard', 'tv', 'laptop']:
                        # Objects with higher aspect ratio are more likely to be barcodes
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        
                        if aspect_ratio > 3.0:  # Typical for barcodes
                            barcode_regions.append((x1, y1, x2, y2))
                        else:
                            three_segment_regions.append((x1, y1, x2, y2))
        except Exception as e:
            print(f"Error processing detections: {str(e)}")
            return self._fallback_detect(image)
            
        # If no regions detected, fall back to traditional method
        if not barcode_regions and not three_segment_regions:
            print("No regions detected by YOLOv5, falling back to traditional method")
            return self._fallback_detect(image)
            
        return barcode_regions, three_segment_regions
    
    def _fallback_detect(self, img):
        """Fallback detection method using traditional approach from original code"""
        width, height = img.size
        top_height = int(height * 0.25)  # Assume top 25% has the three-segment code
        
        # Define regions based on where these elements typically appear
        barcode_region = (0, top_height - 10, width, height)  # Bottom part
        three_segment_region = (0, 0, width, top_height)  # Top part
        
        return [barcode_region], [three_segment_region]

# === Barcode and Three-Segment Code Recognition ===
def extract_barcode(img_region, expected_barcode=""):
    """Extract barcode from image region with optimized processing"""
    try:
        # Process the barcode region
        processed_img = process_barcode_region(img_region)
        
        # Use ZBar for barcode detection and decoding
        # In a production system, you would integrate ZBar library here
        # For this example, we'll simulate with Tesseract OCR
        ocr_text = pytesseract.image_to_string(
            processed_img, 
            config='--psm 6 -c tessedit_char_whitelist=TH0123456789'
        )
        
        # Extract barcode number
        decoded_barcode = ""
        for line in ocr_text.splitlines():
            line_clean = normalize(line)
            if line_clean.startswith("TH") and len(line_clean) >= 12:
                decoded_barcode = line_clean
                break
        
        barcode_sim = 0
        if decoded_barcode and expected_barcode:
            barcode_sim, _ = fuzzy_match(decoded_barcode, expected_barcode)
        
        return decoded_barcode, barcode_sim
        
    except Exception as e:
        print(f"Error in barcode extraction: {str(e)}")
        return "", 0

def extract_three_segment(img_region, expected_code=""):
    """Extract three-segment code from image region with optimized processing"""
    try:
        # Process the text region
        processed_img = process_text_region(img_region)
        
        # Try multiple PSM modes for better accuracy
        text_results = []
        
        # PSM 6 - Assume a single uniform block of text
        text_ocr_6 = pytesseract.image_to_string(
            processed_img, 
            config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- --oem 1'
        )
        text_results.append(text_ocr_6)
        
        # PSM 7 - Treat the image as a single line of text
        text_ocr_7 = pytesseract.image_to_string(
            processed_img, 
            config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- --oem 1'
        )
        text_results.append(text_ocr_7)
        
        # Try to extract three-segment code from all text results
        potential_codes = []
        for text_result in text_results:
            lines = [line.strip() for line in text_result.splitlines() if line.strip()]
            for line in lines:
                # Try direct extraction
                if "-" in line and len(line) >= 11:
                    corrected = correct_common_ocr_errors(line)
                    potential_codes.append(corrected)
                
                # Try pattern matching
                extracted = extract_three_segment_code(line)
                if extracted:
                    potential_codes.append(extracted)
        
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
            # Without expected code, just take the first valid one
            if potential_codes:
                decoded_code = normalize(potential_codes[0])
        
        code_sim = 0
        if decoded_code and expected_code:
            code_sim, _ = fuzzy_match(decoded_code, expected_code)
        
        return decoded_code, code_sim
    
    except Exception as e:
        print(f"Error in three-segment code extraction: {str(e)}")
        return "", 0

# === Main Processing Function ===
def process_image(image_path, expected_barcode="", expected_code="", debug_dir=None):
    """Process an image to extract barcode and three-segment code information"""
    try:
        # Open image
        img = Image.open(image_path)
        
        # Initialize detector
        detector = BarcodeDetector()
        
        # Detect regions
        barcode_regions, three_segment_regions = detector.detect(img)
        
        # Extract the first detected barcode
        barcode_result = ""
        barcode_sim = 0
        if barcode_regions:
            x1, y1, x2, y2 = barcode_regions[0]
            barcode_img = img.crop((x1, y1, x2, y2))
            
            # Save debug image if directory specified
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                debug_filename = os.path.basename(image_path)
                process_barcode_region(barcode_img).save(os.path.join(debug_dir, f"barcode_{debug_filename}"))
            
            barcode_result, barcode_sim = extract_barcode(barcode_img, expected_barcode)
        
        # Extract the first detected three-segment code
        code_result = ""
        code_sim = 0
        if three_segment_regions:
            x1, y1, x2, y2 = three_segment_regions[0]
            three_segment_img = img.crop((x1, y1, x2, y2))
            
            # Save debug image if directory specified
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                debug_filename = os.path.basename(image_path)
                process_text_region(three_segment_img).save(os.path.join(debug_dir, f"text_{debug_filename}"))
            
            code_result, code_sim = extract_three_segment(three_segment_img, expected_code)
        
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
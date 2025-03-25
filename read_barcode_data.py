import csv
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import pytesseract
import difflib
import re
import os
import numpy as np

# === Helper ===
def normalize(text):
    """Normalize text by removing spaces and converting to uppercase"""
    return re.sub(r'\s+', '', text).strip().upper()

def correct_common_ocr_errors(code: str) -> str:
    """Correct common OCR errors in three-segment codes"""
    corrections = {
        '0': 'O',  # 0 → O
        '1': 'I',  # 1 → I
        '5': 'S',  # 5 → S
        '8': 'B',  # 8 → B
        '2': 'Z',  # 2 → Z
        '6': 'G',  # 6 → G
        '|': 'I',  # | → I
        '[': 'I',  # [ → I
        ']': 'I',  # ] → I
        '{': 'I',  # { → I
        '}': 'I',  # } → I
        'l': 'I',  # l → I
        '$': 'S',  # $ → S
        '&': '8',  # & → 8
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

def split_image(img):
    """Split the image into top (three-segment code) and bottom (barcode) regions"""
    width, height = img.size
    top_height = int(height * 0.25)  # Assume top 25% has the three-segment code
    
    top_region = img.crop((0, 0, width, top_height))
    bottom_region = img.crop((0, top_height - 10, width, height))  # Slight overlap
    
    return top_region, bottom_region

def extract_barcode(img, expected_barcode=""):
    """Extract barcode from image with optimized processing"""
    # Process the barcode region
    processed_img = process_barcode_region(img)
    
    # Try multiple PSM modes for better accuracy
    barcode_results = []
    
    # PSM 6 - Assume a single uniform block of text
    ocr_text_6 = pytesseract.image_to_string(
        processed_img, 
        config='--psm 6 -c tessedit_char_whitelist=TH0123456789'
    )
    barcode_results.append(ocr_text_6)
    
    # PSM 7 - Treat the image as a single text line
    ocr_text_7 = pytesseract.image_to_string(
        processed_img, 
        config='--psm 7 -c tessedit_char_whitelist=TH0123456789'
    )
    barcode_results.append(ocr_text_7)
    
    # Extract barcode number
    decoded_barcode = ""
    best_match_ratio = 0
    
    for result in barcode_results:
        for line in result.splitlines():
            line_clean = normalize(line)
            if line_clean.startswith("TH") and len(line_clean) >= 12:
                if expected_barcode:
                    # If we have an expected barcode, check similarity
                    ratio, _ = fuzzy_match(line_clean, expected_barcode, 0)
                    if ratio > best_match_ratio:
                        best_match_ratio = ratio
                        decoded_barcode = line_clean
                else:
                    # Without expected barcode, just take the first valid one
                    decoded_barcode = line_clean
                    break
    
    # If we haven't found a good match but have an expected barcode
    if not decoded_barcode and expected_barcode:
        # Try a more aggressive approach
        threshold_img = processed_img.point(lambda p: 255 if p > 150 else 0)
        ocr_text = pytesseract.image_to_string(
            threshold_img, 
            config='--psm 11 -c tessedit_char_whitelist=TH0123456789'
        )
        
        for line in ocr_text.splitlines():
            line_clean = normalize(line)
            if "TH" in line_clean and len(line_clean) >= 8:
                decoded_barcode = line_clean
                break
    
    barcode_sim = 0
    if decoded_barcode and expected_barcode:
        barcode_sim, _ = fuzzy_match(decoded_barcode, expected_barcode)
    
    return decoded_barcode, barcode_sim

def extract_three_segment(img, expected_code=""):
    """Extract three-segment code from image with optimized processing"""
    # Process the text region
    processed_img = process_text_region(img)
    
    # Try multiple PSM modes to find the best result
    text_results = []
    
    # PSM 6 - Assume a single uniform block of text
    text_ocr_6 = pytesseract.image_to_string(
        processed_img, 
        config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    )
    text_results.append(text_ocr_6)
    
    # PSM 7 - Treat the image as a single text line
    text_ocr_7 = pytesseract.image_to_string(
        processed_img, 
        config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    )
    text_results.append(text_ocr_7)
    
    # PSM 8 - Treat the image as a single word
    text_ocr_8 = pytesseract.image_to_string(
        processed_img, 
        config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    )
    text_results.append(text_ocr_8)
    
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
    
    # If no good match was found, try one more approach
    if best_match_ratio < 0.7 and expected_code:
        # Process with a different technique
        threshold_img = processed_img.point(lambda p: 255 if p > 150 else 0)
        ocr_text = pytesseract.image_to_string(threshold_img, config='--psm 4')
        for line in ocr_text.splitlines():
            if "-" in line:
                corrected = correct_common_ocr_errors(line)
                ratio, _ = fuzzy_match(normalize(corrected), expected_code, 0)
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    decoded_code = normalize(corrected)
    
    code_sim = 0
    if decoded_code and expected_code:
        code_sim, _ = fuzzy_match(decoded_code, expected_code)
    
    return decoded_code, code_sim

# === Config ===
input_csv = "barcode_data.csv"
output_csv = "barcode_result_split.csv"
debug_dir = "ocr_debug"  # For saving debug images

# Create debug directory if needed
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)

# === Main ===
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
        "3Code Diff Count"
    ])

    success_count = 0
    barcode_success_count = 0
    total_count = 0

    for row in reader:
        total_count += 1
        expected_barcode = normalize(row['Tracking Number'])
        expected_code = normalize(row['Three-Code'])
        img_path = row["Filename"]

        try:
            # Open image
            img = Image.open(img_path)
            
            # Split image into regions
            top_region, bottom_region = split_image(img)
            
            # Save debug images
            debug_filename = os.path.basename(img_path)
            
            # ===== STEP 1: First check barcode =====
            decoded_barcode, barcode_sim = extract_barcode(bottom_region, expected_barcode)
            barcode_ok = barcode_sim >= 0.85
            barcode_status = "✅" if barcode_ok else "❌"
            
            if barcode_ok:
                barcode_success_count += 1
                # Save processed bottom region for debugging
                process_barcode_region(bottom_region).save(os.path.join(debug_dir, f"barcode_{debug_filename}"))
            
            # ===== STEP 2: Then check three-segment code =====
            decoded_code, code_sim = extract_three_segment(top_region, expected_code)
            
            # Save processed top region for debugging
            process_text_region(top_region).save(os.path.join(debug_dir, f"text_{debug_filename}"))
            
            # Calculate character differences for three-segment code
            expected_clean = normalize(expected_code)
            decoded_clean = normalize(decoded_code)
            
            # Standardize for comparison
            expected_parts = expected_clean.split('-')
            decoded_parts = decoded_clean.split('-') if '-' in decoded_clean else re.findall(r'.{1,4}', decoded_clean)
            
            if len(expected_parts) >= 3 and len(decoded_parts) >= 3:
                # Compare part by part for more accurate diff count
                diff_count = (
                    char_diff(expected_parts[0], decoded_parts[0][:len(expected_parts[0])]) +
                    char_diff(expected_parts[1], decoded_parts[1][:len(expected_parts[1])]) +
                    char_diff(expected_parts[2], decoded_parts[2][:len(expected_parts[2])])
                )
            else:
                diff_count = char_diff(expected_clean, decoded_clean) if len(decoded_clean) > 0 else 10
            
            # Determine status
            if len(decoded_code) < 3:
                code_status = "⚠️ Missing"
            elif code_sim >= 0.95:
                code_status = "✅"
                success_count += 1
            elif code_sim >= 0.85:
                code_status = "⚠️ Close"
            else:
                code_status = "❌"
            
            writer.writerow([
                img_path,
                expected_barcode,
                expected_code,
                decoded_barcode,
                decoded_code,
                barcode_status,
                code_status,
                f"{barcode_sim:.2f}",
                f"{code_sim:.2f}",
                diff_count
            ])
            
            # Show both barcode and three-segment code status
            barcode_icon = "✅" if barcode_status == "✅" else "❌"
            code_icon = "✅" if code_status == "✅" else "❌"
            
            print(f"[{barcode_icon}|{code_icon}] {os.path.basename(img_path)}: {expected_barcode} → {decoded_barcode} | {expected_code} → {decoded_code}")

        except Exception as e:
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
                "10"
            ])
            print(f"[⚠️] {img_path} → {str(e)}")

    # Calculate success rates
    barcode_success_rate = (barcode_success_count / total_count) * 100 if total_count > 0 else 0
    three_code_success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"\n📄 วิเคราะห์และ export เสร็จ: {output_csv} ✅")
    print(f"📊 อัตราความสำเร็จ Barcode: {barcode_success_count}/{total_count} ({barcode_success_rate:.1f}%)")
    print(f"📊 อัตราความสำเร็จ Three-Segment Code: {success_count}/{total_count} ({three_code_success_rate:.1f}%)")
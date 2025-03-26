import os
import cv2
from pathlib import Path
import csv
import time
import re

# Import functions from read_barcode_data.py
from read_barcode_data import (
    process_image,
    normalize,
    fuzzy_match,
    enhance_image,
    recognize_barcode,
    recognize_three_segment_code,
    split_image
)

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def process_labeled_images(input_folder, output_csv, barcode_data_csv, debug_dir=None):
    """
    Process images listed in barcode_data.csv:
    1. Read expected barcode and three-segment code from CSV
    2. Read the corresponding image file
    3. Extract barcode and three-segment code using OCR
    4. Compare with expected values
    5. Save results to CSV
    
    Args:
        input_folder: Folder containing the detected label images
        output_csv: Path to save results CSV
        barcode_data_csv: CSV file with expected barcode and code values
        debug_dir: Optional directory for saving debug images
    """
    # Create debug directory if needed
    if debug_dir:
        create_directory(debug_dir)
    
    # Read barcode data CSV
    data_rows = []
    try:
        with open(barcode_data_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data_rows.append({
                    'filename': row['Filename'],
                    'expected_barcode': normalize(row['Tracking Number']),
                    'expected_code': normalize(row['Three-Code'])
                })
        print(f"Loaded {len(data_rows)} entries from {barcode_data_csv}")
    except Exception as e:
        print(f"Error reading barcode data CSV: {str(e)}")
        return
    
    # Create results CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Filename",
            "Expected Barcode",
            "Expected 3Code",
            "Detected Barcode",
            "Detected 3Code",
            "Barcode Status",
            "3Code Status",
            "Barcode Similarity",
            "3Code Similarity",
            "Processing Time (s)"
        ])
        
        # Process each entry
        success_count = 0
        barcode_success_count = 0
        code_success_count = 0
        
        for row in data_rows:
            filename = row['filename']
            expected_barcode = row['expected_barcode']
            expected_code = row['expected_code']
            
            # Construct image path
            img_path = os.path.join(input_folder, filename)
            
            # Skip if file doesn't exist
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                writer.writerow([
                    filename,
                    expected_barcode,
                    expected_code,
                    "⚠️ File not found",
                    "⚠️ File not found",
                    "❌",
                    "❌",
                    "0.00",
                    "0.00",
                    "0.00"
                ])
                continue
            
            # Process the image
            start_time = time.time()
            
            try:
                # Read the image
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Could not read image: {img_path}")
                
                # Process image with OCR to extract barcode and three-segment code
                label_debug_dir = os.path.join(debug_dir, os.path.splitext(filename)[0]) if debug_dir else None
                result = process_image(
                    img_path, 
                    expected_barcode=expected_barcode,
                    expected_code=expected_code,
                    debug_dir=label_debug_dir
                )
                
                processing_time = time.time() - start_time
                
                # Determine success
                barcode_ok = result["barcode_status"] == "✅"
                code_ok = result["code_status"] == "✅"
                
                if barcode_ok:
                    barcode_success_count += 1
                if code_ok:
                    code_success_count += 1
                if barcode_ok and code_ok:
                    success_count += 1
                
                # Write to CSV
                writer.writerow([
                    filename,
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
                
                # Log to console
                print(f"[{result['barcode_status']}|{result['code_status']}] {filename}: {expected_barcode} → {result['barcode']} | {expected_code} → {result['three_segment_code']}")
                
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"Error processing {filename}: {str(e)}")
                writer.writerow([
                    filename,
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
        
        # Calculate success rates
        total = len(data_rows)
        barcode_rate = (barcode_success_count / total) * 100 if total > 0 else 0
        code_rate = (code_success_count / total) * 100 if total > 0 else 0
        both_rate = (success_count / total) * 100 if total > 0 else 0
        
        print(f"\n📄 Analysis complete: {output_csv} ✅")
        print(f"📊 Processed {total} images")
        print(f"📊 Barcode Success Rate: {barcode_success_count}/{total} ({barcode_rate:.1f}%)")
        print(f"📊 Three-Segment Code Success Rate: {code_success_count}/{total} ({code_rate:.1f}%)")
        print(f"📊 Overall Success Rate: {success_count}/{total} ({both_rate:.1f}%)")

def find_missed_images(input_folder, barcode_data_csv, missed_output_csv):
    """
    Find images that were missed during YOLO detection.
    Compare entries in barcode_data.csv with files in input_folder.
    
    Args:
        input_folder: Folder containing detected label images
        barcode_data_csv: CSV with expected data
        missed_output_csv: CSV to save missed images
    """
    # Read barcode data CSV
    expected_files = set()
    try:
        with open(barcode_data_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                expected_files.add(row['Filename'])
    except Exception as e:
        print(f"Error reading barcode data CSV: {str(e)}")
        return
    
    # Get files in input folder
    actual_files = set()
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        for file in Path(input_folder).glob(f"*{ext}"):
            actual_files.add(file.name)
        for file in Path(input_folder).glob(f"*{ext.upper()}"):
            actual_files.add(file.name)
    
    # Find missing files
    missing_files = expected_files - actual_files
    
    # Save to CSV
    with open(missed_output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Missing Filename"])
        for filename in sorted(missing_files):
            writer.writerow([filename])
    
    print(f"Found {len(missing_files)} files listed in {barcode_data_csv} but missing from {input_folder}")
    print(f"Missing files saved to: {missed_output_csv}")

if __name__ == "__main__":
    # Configuration
    input_folder = "label"  # Folder containing detected label images
    output_csv = "barcode_result_improved.csv"  # Results CSV (matching the name in read_barcode_data.py)
    barcode_data_csv = "barcode_data.csv"  # CSV with expected values
    debug_dir = "ocr_debug_improved"  # Debug images directory (matching the name in read_barcode_data.py)
    missed_output_csv = "missing_files.csv"  # CSV for missing files
    
    # Step 1: Find any missed images (files in CSV but not in folder)
    find_missed_images(input_folder, barcode_data_csv, missed_output_csv)
    
    # Step 2: Process detected images
    process_labeled_images(
        input_folder=input_folder,
        output_csv=output_csv,
        barcode_data_csv=barcode_data_csv,
        debug_dir=debug_dir
    )
import csv
from PIL import Image
import pytesseract
import re
import difflib

def normalize_barcode(text):
    text = text.replace(']', '|').replace('/', '|')  # เพิ่มตรงนี้
    text = re.sub(r'\s*\|\s*', '|', text)
    text = re.sub(r'\s+', '', text)  # remove all spaces
    return text.strip()

def fuzzy_match(a, b, threshold=0.95):
    return difflib.SequenceMatcher(None, a, b).ratio(), difflib.SequenceMatcher(None, a, b).ratio() >= threshold

input_csv = "barcode_data.csv"
output_csv = "barcode_result_split.csv"

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
        "Similarity 3Code"
    ])

    for row in reader:
        expected_barcode = f"{row['Tracking Number']}|{row['Sender']}|{row['Receiver']}|{row['Date']}"
        expected_code = row['Three-Code']
        expected_barcode_norm = normalize_barcode(expected_barcode)
        expected_code_norm = expected_code.strip()

        img_path = row["Filename"]

        try:
            img = Image.open(img_path)
            ocr_text = pytesseract.image_to_string(img, config='--psm 6').strip()
            lines = [line.strip() for line in ocr_text.splitlines() if '|' in line or ']' in line]
            decoded_line = lines[0] if lines else ""

            decoded_line = decoded_line.replace(']', '|')
            decoded_line = normalize_barcode(decoded_line)

            if decoded_line.count("|") >= 4:
                parts = decoded_line.split("|")
                decoded_barcode = "|".join(parts[:4])
                decoded_code = parts[4]
            else:
                decoded_barcode = decoded_line
                decoded_code = ""

            # Match check
            barcode_sim, barcode_ok = fuzzy_match(decoded_barcode, expected_barcode_norm)
            code_sim, code_ok = fuzzy_match(decoded_code, expected_code_norm)

            writer.writerow([
                img_path,
                expected_barcode_norm,
                expected_code_norm,
                decoded_barcode,
                decoded_code,
                "✅" if barcode_ok else "❌",
                "✅" if code_ok else "❌",
                f"{barcode_sim:.2f}",
                f"{code_sim:.2f}"
            ])

            print(f"[Barcode:{'✅' if barcode_ok else '❌'} | Code:{'✅' if code_ok else '❌'}] {img_path}")

        except Exception as e:
            writer.writerow([
                img_path,
                expected_barcode_norm,
                expected_code_norm,
                "⚠️ Error",
                "⚠️ Error",
                "❌",
                "❌",
                "0.00",
                "0.00"
            ])
            print(f"[⚠️] {img_path} → {e}")

print("\n📄 Export สำเร็จ: barcode_result_split.csv ✅")

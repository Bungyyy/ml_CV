import csv
from PIL import Image
import pytesseract
import re
import difflib

# === Helper ===
def normalize_barcode(text):
    text = text.replace(']', '|').replace('/', '|')
    text = re.sub(r'\s*\|\s*', '|', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()

def correct_common_ocr_errors(code: str) -> str:
    return (
        code.upper()
            .replace('0', 'O')
            .replace('1', 'I')
            .replace('5', 'S')
            .replace('8', 'B')
    )

def fuzzy_match(a, b, threshold=0.95):
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio, ratio >= threshold

def char_diff(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)

# === Config ===
input_csv = "barcode_data.csv"
output_csv = "barcode_result_split.csv"

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

    for row in reader:
        expected_barcode = f"{row['Tracking Number']}|{row['Sender']}|{row['Receiver']}|{row['Date']}"
        expected_code = row['Three-Code']

        expected_barcode_norm = normalize_barcode(expected_barcode)
        expected_code_norm = correct_common_ocr_errors(expected_code.strip())

        img_path = row["Filename"]

        try:
            img = Image.open(img_path)
            ocr_text = pytesseract.image_to_string(img, config='--psm 6').strip()
            lines = [line.strip() for line in ocr_text.splitlines() if '|' in line or ']' in line or '/' in line]
            decoded_line = lines[0] if lines else ""

            decoded_line = normalize_barcode(decoded_line)

            if decoded_line.count("|") >= 4:
                parts = decoded_line.split("|")
                decoded_barcode = "|".join(parts[:4])
                decoded_code = correct_common_ocr_errors(parts[4])
            else:
                decoded_barcode = decoded_line
                decoded_code = ""

            # Match barcode
            barcode_sim, barcode_ok = fuzzy_match(decoded_barcode, expected_barcode_norm)
            barcode_status = "✅" if barcode_ok else "❌"

            # Match 3-code
            code_sim, code_ok = fuzzy_match(decoded_code, expected_code_norm)
            diff_count = char_diff(decoded_code, expected_code_norm) if len(decoded_code) == 3 else 3

            # ตรวจสอบความยาวขาด
            if len(decoded_code) < 3:
                code_status = "⚠️ Missing"
            elif code_sim == 1.0:
                code_status = "✅"
            elif diff_count == 1:
                code_status = "⚠️"
            else:
                code_status = "❌"

            writer.writerow([
                img_path,
                expected_barcode_norm,
                expected_code_norm,
                decoded_barcode,
                decoded_code,
                barcode_status,
                code_status,
                f"{barcode_sim:.2f}",
                f"{code_sim:.2f}",
                diff_count
            ])

            print(f"[Barcode:{barcode_status} | Code:{code_status}] {img_path}")

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
                "0.00",
                "3"
            ])
            print(f"[⚠️] {img_path} → {e}")

print("\n📄 วิเคราะห์และ export เสร็จ: barcode_result_split.csv ✅")

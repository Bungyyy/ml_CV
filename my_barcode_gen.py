import barcode
from barcode.writer import ImageWriter
from datetime import datetime, timedelta
import random
import os
import csv
from PIL import Image, ImageDraw, ImageFont

# เตรียมโฟลเดอร์เก็บบาร์โค้ด
output_dir = "barcodes"
os.makedirs(output_dir, exist_ok=True)

csv_filename = "barcode_data.csv"

# Font (ใช้ font มาตรฐานของ macOS หรือใส่ path เอง)
try:
    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 48)
except:
    font = ImageFont.load_default()

senders = [
    "Rattapol", "Somchai", "Anya", "Nattapon", "Suda", "Ploy", "Junior",
    "Petch", "Beam", "Mild", "May", "Fah", "Game", "Oil", "Mind",
    "Tan", "Fon", "Nat", "Belle", "Nene", "Tle", "Nook", "Golf",
    "Jub", "Jin", "View", "Ice", "Oat", "Pang", "Mook"
]
receivers = [
    "Boss", "Nina", "Mek", "Kwan", "Jay", "Gift",
    "Bank", "Nam", "Palm", "Aom", "Toon", "Best", "Top", "Fern", "Tarn",
    "Benz", "Zee", "Pangpond", "Atom", "Mint", "Yui", "Ying", "Dream",
    "Bam", "Noon", "Gade", "Tukta", "Oom", "Waan", "Cake"
]

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Tracking Number", "Sender", "Receiver", "Date", "Three-Code", "Filename"])

    for i in range(1, 201):
        tracking_number = f"TH{str(1000000000 + i)}"
        sender = random.choice(senders)
        receiver = random.choice(receivers)
        date = (datetime.today() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        three_code = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=3))

        data = f"{tracking_number}|{sender}|{receiver}|{date}|{three_code}"

        barcode_class = barcode.get_barcode_class('code128')
        barcode_obj = barcode_class(data, writer=ImageWriter())

        # สร้างไฟล์ชั่วคราวของ barcode
        temp_path = os.path.join(output_dir, f"{tracking_number}_raw")
        barcode_obj.save(temp_path)
        temp_filename = f"{temp_path}.png"  # <- เพิ่มหลังจาก save แล้วใช้ชื่อไฟล์ที่ถูกต้อง

        # เปิดภาพ barcode ที่สร้างไว้
        barcode_img = Image.open(temp_filename)
        width, height = barcode_img.size

        # สร้างภาพใหม่ + พื้นที่ด้านบน
        final_height = height + 40
        final_img = Image.new("RGB", (width, final_height), "white")

        # วาดข้อความ 3 ตัวด้านบน
        draw = ImageDraw.Draw(final_img)
        text_width, _ = draw.textsize(three_code, font=font)
        draw.text(((width - text_width) / 2, 20), three_code, font=font, fill="black")  # ขยับลง

        # แปะ barcode ลงด้านล่าง
        final_img.paste(barcode_img, (0, 70))

        # บันทึกภาพสุดท้าย
        final_filename = os.path.join(output_dir, f"{tracking_number}.png")
        final_img.save(final_filename)

        # ลบภาพชั่วคราว
        os.remove(temp_filename)

        # เขียน CSV
        writer.writerow([tracking_number, sender, receiver, date, three_code, final_filename])
        print(f"[{i}/200] ✅ สร้างแล้ว: {final_filename}")

print(f"\n📄 CSV export สำเร็จ: {csv_filename}")

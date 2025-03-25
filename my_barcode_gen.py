import barcode
from barcode.writer import ImageWriter
import random
import os
import csv
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

# โฟลเดอร์ output
output_dir = "barcodes"
os.makedirs(output_dir, exist_ok=True)

csv_filename = "barcode_data.csv"

    # ใช้ฟอนต์ที่เป็นมิตรกับ OCR และแสดงตัวอักษร Q เต็มตัว
try:
    # ลองหาฟอนต์ที่เหมาะสม โดยเรียงลำดับจากฟอนต์ที่แสดงตัวอักษร Q ได้ชัดเจนที่สุด
    ocr_fonts = ["Arial", "Tahoma", "Verdana", "Helvetica", "DejaVu Sans", "Liberation Sans", 
                "DejaVu Sans Mono", "Liberation Mono", "Courier New", "OCR-A", "OCR-B"]
    font_path = None
    
    for font_name in ocr_fonts:
        try:
            font_path = fm.findfont(font_name)
            if font_path:
                print(f"ใช้ฟอนต์: {font_name}")
                break
        except:
            continue
    
    if font_path:
        # ขนาดฟอนต์ที่เหมาะสมสำหรับ three-segment code
        font = ImageFont.truetype(font_path, 80)
    else:
        print("ไม่พบฟอนต์ที่ต้องการ ใช้ฟอนต์เริ่มต้นแทน")
        font = ImageFont.load_default()
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดฟอนต์: {e}")
    font = ImageFont.load_default()

# ฟังก์ชันสร้าง 3-segment code ที่เป็นมิตรกับ OCR
def gen_three_segment_code():
    # ใช้เฉพาะตัวอักษรที่ OCR แยกแยะได้ง่าย - ยังคงมี Q แต่จะใช้ฟอนต์ที่แสดง Q เต็มตัว
    ocr_friendly_chars = 'ABCDEFGHJKLMNPQRSTUVWXY3456789'
    part1 = ''.join(random.choices(ocr_friendly_chars, k=3))
    part2 = ''.join(random.choices(ocr_friendly_chars, k=4))
    part3 = ''.join(random.choices(ocr_friendly_chars, k=3))
    # ใช้ dash สั้นแทนเพื่อให้ตัวอักษรชิดกันมากขึ้น
    return f"{part1} - {part2} - {part3}"

# สร้าง CSV
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Tracking Number", "Three-Code", "Filename"])

    for i in range(1, 201):
        try:
            tracking_number = f"TH{1000000000 + i}"
            three_code = gen_three_segment_code()

            # ใช้ barcode จาก tracking number
            barcode_class = barcode.get_barcode_class('code128')
            barcode_obj = barcode_class(tracking_number, writer=ImageWriter())

            # บันทึก barcode แบบไม่มีสาม segment code
            temp_path = os.path.join(output_dir, f"{tracking_number}_raw")
            temp_filename = f"{temp_path}.png"

            # ปรับแต่งค่าของ barcode ให้ชัดเจน
            barcode_options = {
                "module_width": 0.6,      # ความหนาของแท่ง barcode
                "module_height": 30,      # ความสูงของแท่ง barcode
                "font_size": 18,          # ขนาดฟอนต์ของตัวเลขใต้ barcode
                "text_distance": 7,       # ระยะห่างระหว่างแท่ง barcode กับตัวเลข
                "quiet_zone": 6,          # พื้นที่ว่างรอบ barcode
                "write_text": True        # แสดงตัวเลขใต้ barcode
            }
            
            barcode_obj.save(temp_path, options=barcode_options)

            # เปิดภาพ barcode
            barcode_img = Image.open(temp_filename)
            width, height = barcode_img.size

            # เพิ่มพื้นที่ด้านบนสำหรับ three-segment code
            padding_top = 100  # พื้นที่ด้านบน
            padding_side = 40  # พื้นที่ด้านข้าง
            
            final_width = width + padding_side * 2
            final_height = height + padding_top
            
            # สร้างภาพใหม่ด้วยพื้นหลังสีขาว
            final_img = Image.new("RGB", (final_width, final_height), (255, 255, 255))
            draw = ImageDraw.Draw(final_img)
            
            # วัดขนาดข้อความ three-code
            try:
                # สำหรับ PIL เวอร์ชันใหม่
                bbox = draw.textbbox((0, 0), three_code, font=font)
                text_width = bbox[2] - bbox[0]
            except AttributeError:
                # สำหรับ PIL เวอร์ชันเก่า
                try:
                    text_width, _ = draw.textsize(three_code, font=font)
                except:
                    text_width = width * 0.8  # ประมาณค่า
            
            # วาด three-code ด้านบนตรงกลาง
            text_x = (final_width - text_width) / 2
            text_y = 20  # ระยะจากด้านบน
            draw.text((text_x, text_y), three_code, font=font, fill=(0, 0, 0))
            
            # แปะ barcode ด้านล่าง (ไม่มีเส้นคั่น)
            barcode_x = padding_side
            barcode_y = text_y + 80
            final_img.paste(barcode_img, (barcode_x, barcode_y))

            # บันทึกภาพสุดท้าย
            final_filename = os.path.join(output_dir, f"{tracking_number}.png")
            final_img.save(final_filename, dpi=(300, 300))

            # ลบไฟล์ชั่วคราว
            try:
                os.remove(temp_filename)
            except:
                print(f"ไม่สามารถลบไฟล์ชั่วคราว: {temp_filename}")

            # บันทึกลง CSV
            writer.writerow([tracking_number, three_code, final_filename])
            print(f"[{i}/200] ✅ สร้างแล้ว: {final_filename}")
            
        except Exception as e:
            print(f"[{i}/200] ❌ เกิดข้อผิดพลาด: {str(e)}")

print(f"\n📄 สร้าง CSV เรียบร้อย: {csv_filename}")
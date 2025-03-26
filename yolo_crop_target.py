import os
import cv2
from pathlib import Path
from ultralytics import YOLO

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def crop_objects(input_folder, output_folder, model_path, target_class="Target"):
    """
    Process all images in a folder, detect objects labeled with the target class,
    crop them, and save to output folder.
    """
    # Create output directory
    create_directory(output_folder)
    
    # Load YOLOv11 model
    model = YOLO(model_path)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f"*{ext}")))
        image_files.extend(list(Path(input_folder).glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    target_count = 0
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        print(f"Processing: {img_name}")
        
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
            
            target_count += 1
            print(f"Saved: {output_name}")
    
    print(f"Processing complete. Found and saved {target_count} '{target_class}' objects.")

# Example usage
# Corrected function call
if __name__ == "__main__":
    input_folder = "label"
    output_folder = "output"
    model_path = "best.pt"
    target_class = "Target"

    crop_objects(input_folder, output_folder, model_path=model_path, target_class=target_class)
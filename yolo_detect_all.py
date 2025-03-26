import os
import cv2
from ultralytics import YOLO

def detect_and_save_images(input_folder, output_folder, model_path):
    """
    Detect objects in all images in a folder using YOLOv11 and save the results.
    
    Parameters:
    - input_folder: Path to folder containing input images
    - output_folder: Path to folder for saving output images
    - model_path: Path to model weights
    """
    # Load model
    model = YOLO(model_path)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files in the input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    # Process each image
    for image_file in image_files:
        # Construct full paths
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        
        # Perform prediction
        results = model.predict(input_path)
        
        # Access first result
        result = results[0]
        
        # Save with bounding boxes
        result.save(filename=output_path)
        
        print(f"Processed: {image_file}")

# Example usage
if __name__ == "__main__":
    input_folder = "label"  # Folder containing images
    output_folder = "output"  # Folder to save results
    model_path = "best.pt"  # Path to model weights
    detect_and_save_images(input_folder, output_folder, model_path)
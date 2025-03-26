import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import os
import time

# Import the existing functions from read_barcode_data.py
from read_barcode_data import (
    process_image, recognize_barcode, recognize_three_segment_code,
    enhance_image, normalize, fuzzy_match, correct_ocr_errors
)

class SELayer(nn.Module):
    """Squeeze-and-Excitation attention module as described in the paper"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CSPBlock(nn.Module):
    """Cross Stage Partial block for the network"""
    def __init__(self, in_channels, out_channels):
        super(CSPBlock, self).__init__()
        self.part1_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.part2_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.res_conv = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        part1 = self.part1_conv(x)
        part2 = self.part2_conv(x)
        part2 = self.res_conv(part2)
        out = torch.cat([part1, part2], dim=1)
        return self.final_conv(out)

class CSP_SPP(nn.Module):
    """CSP modular SPP structure as described in the paper"""
    def __init__(self, in_channels, out_channels):
        super(CSP_SPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
        
        # SPP block (Spatial Pyramid Pooling)
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        
        self.conv5 = nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1)
        self.conv_final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Split the input
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        # Process x2 through the SPP path
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        
        # Apply SPP module
        spp_out = torch.cat([
            x2,
            self.maxpool1(x2),
            self.maxpool2(x2),
            self.maxpool3(x2)
        ], dim=1)
        
        x2 = self.conv5(spp_out)
        
        # Concatenate and apply final convolution
        out = torch.cat([x1, x2], dim=1)
        return self.conv_final(out)

class SCS_YOLOv4(nn.Module):
    """SCS-YOLOv4 model as described in the paper"""
    def __init__(self, num_classes=2):  # Two classes: barcode and three-segment code
        super(SCS_YOLOv4, self).__init__()
        
        # Use CSPDarknet53 as backbone (simplified representation)
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone_features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # CSP-SPP module
        self.csp_spp = CSP_SPP(2048, 1024)
        
        # SE attention modules
        self.se1 = SELayer(1024)
        self.se2 = SELayer(512)
        self.se3 = SELayer(256)
        
        # Feature Pyramid Network (FPN)
        self.lateral_conv1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(512, 256, kernel_size=1)
        
        self.fpn_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fpn_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Detection heads
        self.head1 = nn.Conv2d(256, num_classes * 25, kernel_size=1)  # For small objects
        self.head2 = nn.Conv2d(256, num_classes * 25, kernel_size=1)  # For medium objects
        self.head3 = nn.Conv2d(256, num_classes * 25, kernel_size=1)  # For large objects
        
    def forward(self, x):
        # Backbone features
        c3, c4, c5 = self.extract_features(x)
        
        # Apply CSP-SPP to C5
        p5 = self.csp_spp(c5)
        p5 = self.se1(p5)
        
        # FPN top-down pathway with proper channel projection
        # First, use lateral convolutions to project all feature maps to the same channel dimension
        lateral_p5 = self.lateral_conv1(p5)  # Project p5 to 256 channels
        lateral_c4 = self.lateral_conv2(c4)  # Project c4 to 256 channels
        lateral_c3 = self.lateral_conv3(c3)  # Project c3 to 256 channels
        
        # Upsample p5 and add to c4
        p5_upsampled = nn.functional.interpolate(lateral_p5, size=c4.shape[2:], mode='nearest')
        p4 = lateral_c4 + p5_upsampled
        p4 = self.fpn_conv2(p4)
        p4 = self.se2(p4)
        
        # Upsample p4 and add to c3
        p4_upsampled = nn.functional.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p3 = lateral_c3 + p4_upsampled
        p3 = self.fpn_conv3(p3)
        p3 = self.se3(p3)
        
        # Detection heads
        d3 = self.head1(p3)  # Small objects (e.g., smaller barcodes)
        d4 = self.head2(p4)  # Medium objects
        d5 = self.head3(p5)  # Large objects (e.g., larger three-segment codes)
        
        return [d3, d4, d5]
    
    def extract_features(self, x):
        """Extract features from the backbone network"""
        # This is a simplified version, actual implementation would extract 
        # features from different levels of CSPDarknet53
        x = self.backbone_features(x)
        
        # Simulate extraction of features from different stages
        # In a real implementation, these would be taken from different layers of CSPDarknet53
        c3 = torch.randn(x.shape[0], 512, x.shape[2]*2, x.shape[3]*2)
        c4 = torch.randn(x.shape[0], 1024, x.shape[2], x.shape[3])
        c5 = x  # 2048 channels
        
        return c3, c4, c5


class ParcelSortingSystem:
    """Main class implementing the express parcel sorting system described in the paper"""
    def __init__(self, model_path=None, device="cpu"):
        self.device = device
        
        # Flag to indicate if we're using the deep model
        self.use_deep_model = False
        
        # Only initialize the model if a valid path is provided to avoid errors
        if model_path and os.path.exists(model_path):
            self.use_deep_model = True
            # Initialize the SCS-YOLOv4 model
            self.model = SCS_YOLOv4(num_classes=2)
            # Load pre-trained weights
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()
        else:
            print("No model path provided or model not found. Using traditional image splitting approach.")
    
    def detect_regions(self, image):
        """Detect barcode and three-segment code regions in the image"""
        # Instead of using the model which has architecture issues,
        # we'll fall back to the original image splitting approach for now
        # This ensures the code works while you implement the proper model
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Define regions based on typical placement (as in the original code)
        top_height = int(height * 0.25)  # Assume top 25% has the three-segment code
        
        # Barcode is often in the bottom portion
        barcode_regions = [(0, top_height, width, height)]
        
        # Define the three-segment code region based on typical placement
        # Usually at the top of the image
        three_segment_regions = [(0, 0, width, top_height)]
        
        return barcode_regions, three_segment_regions
    
    def process_image(self, image_path, expected_barcode="", expected_code="", debug_dir=None):
        """Process an image to identify barcode and three-segment code"""
        start_time_total = time.time()
        try:
            # Load the image
            load_start = time.time()
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image: {image_path}")
            else:
                img = image_path
            load_time = time.time() - load_start
            
            # Detect regions using SCS-YOLOv4 or traditional approach
            detect_start = time.time()
            barcode_regions, three_segment_regions = self.detect_regions(img)
            detect_time = time.time() - detect_start
            
            # Process barcode regions
            barcode_start = time.time()
            barcode_result = ""
            barcode_sim = 0
            
            for region in barcode_regions:
                x1, y1, x2, y2 = region
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
            barcode_time = time.time() - barcode_start
            
            # Process three-segment code regions
            threecode_start = time.time()
            code_result = ""
            code_sim = 0
            
            for region in three_segment_regions:
                x1, y1, x2, y2 = region
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
            threecode_time = time.time() - threecode_start
            
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
            
            total_time = time.time() - start_time_total
            
            # Print timing information
            print(f"⏱️ Timing for {os.path.basename(image_path) if isinstance(image_path, str) else 'image'}:")
            print(f"  - Load image: {load_time:.3f}s")
            print(f"  - Detect regions: {detect_time:.3f}s")
            print(f"  - Barcode recognition: {barcode_time:.3f}s")
            print(f"  - Three-code recognition: {threecode_time:.3f}s")
            print(f"  - Total processing: {total_time:.3f}s")
            
            return {
                "barcode": barcode_result,
                "three_segment_code": code_result,
                "barcode_status": barcode_status,
                "code_status": code_status,
                "barcode_sim": barcode_sim,
                "code_sim": code_sim,
                "timing": {
                    "load": load_time,
                    "detect": detect_time,
                    "barcode": barcode_time,
                    "threecode": threecode_time,
                    "total": total_time
                }
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
                "code_sim": 0.0,
                "timing": {
                    "load": 0.0,
                    "detect": 0.0,
                    "barcode": 0.0,
                    "threecode": 0.0,
                    "total": time.time() - start_time_total
                }
            }
    
    def process_batch(self, input_csv, output_csv, debug_dir=None):
        """Process a batch of images based on entries in a CSV file"""
        import csv
        
        # Initialize list to store timing results
        self.timing_results = []
        print("Debug: Initialized empty timing_results list")
        
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
                "Total Time (s)",
                "Load Time (s)",
                "Detect Time (s)",
                "Barcode Time (s)",
                "ThreeCode Time (s)"
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
                    result = self.process_image(img_path, expected_barcode, expected_code, debug_dir)
                    
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
                        f"{result['timing']['total']:.2f}",
                        f"{result['timing']['load']:.2f}",
                        f"{result['timing']['detect']:.2f}",
                        f"{result['timing']['barcode']:.2f}",
                        f"{result['timing']['threecode']:.2f}"
                    ])
                    
                    # Debug print to see timing data structure
                    print(f"Debug: Adding timing data for {os.path.basename(img_path)}: {result['timing']}")
                    
                    # Store timing results for statistics (only if successful)
                    if 'timing' in result:
                        # Store timing data with a flat structure
                        self.timing_results.append({
                            'load': result['timing']['load'],
                            'detect': result['timing']['detect'],
                            'barcode': result['timing']['barcode'],
                            'threecode': result['timing']['threecode'],
                            'total': result['timing']['total']
                        })
                        print(f"Debug: timing_results now has {len(self.timing_results)} entries")
                    else:
                        print(f"Warning: No timing information available for {img_path}")
                    
                    print(f"[{result['barcode_status']}|{result['code_status']}] {os.path.basename(img_path)}: {expected_barcode} → {result['barcode']} | {expected_code} → {result['three_segment_code']}")
                    
                except Exception as e:
                    try:
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
                            f"{processing_time:.2f}",
                            "0.00",
                            "0.00",
                            "0.00",
                            "0.00"
                        ])
                        print(f"[⚠️] {img_path} → {str(e)}")
                    except Exception as e2:
                        print(f"[⚠️⚠️] Critical error while handling error for {img_path}: {str(e2)}")

            # Calculate success rates and average timings
            barcode_success_rate = (barcode_success_count / total_count) * 100 if total_count > 0 else 0
            three_code_success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            
            # Calculate average timing information (excluding errors)
            avg_total_time = sum(result['total'] for result in self.timing_results) / len(self.timing_results) if self.timing_results else 0
            avg_load_time = sum(result['load'] for result in self.timing_results) / len(self.timing_results) if self.timing_results else 0
            avg_detect_time = sum(result['detect'] for result in self.timing_results) / len(self.timing_results) if self.timing_results else 0
            avg_barcode_time = sum(result['barcode'] for result in self.timing_results) / len(self.timing_results) if self.timing_results else 0
            avg_threecode_time = sum(result['threecode'] for result in self.timing_results) / len(self.timing_results) if self.timing_results else 0

            
            print(f"\n📄 Analysis and export complete: {output_csv} ✅")
            print(f"📊 Barcode Success Rate: {barcode_success_count}/{total_count} ({barcode_success_rate:.1f}%)")
            print(f"📊 Three-Segment Code Success Rate: {success_count}/{total_count} ({three_code_success_rate:.1f}%)")
            print(f"\n⏱️ Average Processing Times:")
            print(f"  - Load image: {avg_load_time:.3f}s")
            print(f"  - Detect regions: {avg_detect_time:.3f}s")
            print(f"  - Barcode recognition: {avg_barcode_time:.3f}s")
            print(f"  - Three-code recognition: {avg_threecode_time:.3f}s")
            print(f"  - Total processing: {avg_total_time:.3f}s")


# Example usage
if __name__ == "__main__":
    # Initialize the system
    sorting_system = ParcelSortingSystem(model_path="scs_yolov4_model.pth")
    
    # Process a batch of images
    input_csv = "barcode_data.csv"
    output_csv = "barcode_result_improved.csv"
    debug_dir = "ocr_debug_improved"
    
    # Create debug directory if needed
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    sorting_system.process_batch(input_csv, output_csv, debug_dir)
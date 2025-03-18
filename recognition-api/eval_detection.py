import os
import cv2
from ultralytics import YOLO
import asyncio

from utils.license_plate_detection import detect_license_plate
from utils.section_detection import detect_sections

YOLO_DIR = os.path.join("recognition-api/model/yolo")

def load_eval_dataset(eval_dataset_path):
    images = []
    for root, _, files in os.walk(eval_dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                images.append(image_path)
    return images

def save_annotated_image(image, detections, sections=None, filename="", is_filtered=False):
    annotated = image.copy()
    
    # Draw license plate detection
    for x1, y1, x2, y2 in detections:
        color = (0, 0, 255) if is_filtered else (255, 0, 0)  # Red for filtered, Blue for original
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw sections if available
        if sections:
            for cls_number, s_x1, s_y1, s_x2, s_y2 in sections:
                cv2.rectangle(annotated, 
                            (x1+s_x1, y1+s_y1), 
                            (x1+s_x2, y1+s_y2), 
                            (0, 255, 0), 2)
                # Add section type label
                section_type = cls_.get(cls_number, "unknown")
                cv2.putText(annotated, 
                          section_type, 
                          (x1+s_x1, y1+s_y1-5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (0, 255, 0),
                          1)
    
    output_dir = "detection_results"
    os.makedirs(output_dir, exist_ok=True)
    prefix = "filtered_" if is_filtered else "annotated_"
    cv2.imwrite(os.path.join(output_dir, f"{prefix}{filename}"), annotated)

def validate_section_positions(sections):
    """Validate if detected sections are in correct positions relative to each other."""
    # First check: exactly 4 unique sections
    if not sections or len(sections) != 4:
        return False, "Need exactly 4 sections"
    
    # Check for uniqueness
    section_numbers = set(s[0] for s in sections)
    if len(section_numbers) != 4:
        return False, "Duplicate sections detected"
    
    # Create a dictionary of sections by their class number
    section_dict = {}
    for cls_number, x1, y1, x2, y2 in sections:
        section_dict[cls_number] = {
            'x': (x1 + x2) / 2,  # center x
            'y': (y1 + y2) / 2,  # center y
        }
    
    # Check if all required sections (0-3) are present
    if not all(i in section_dict for i in range(4)):
        return False, "Missing one or more required sections"
    
    # Validate relative positions
    # region(0) = top-left
    # classification(2) = top-right
    # hiragana(1) = bottom-left
    # number(3) = bottom-right
    
    region = section_dict[0]
    hiragana = section_dict[1]
    classification = section_dict[2]
    number = section_dict[3]
    
    errors = []
    
    # Horizontal position checks
    if region['x'] >= classification['x']:
        errors.append("Region must be strictly left of classification")
    if hiragana['x'] >= number['x']:
        errors.append("Hiragana must be strictly left of number")
    
    # Vertical position checks
    if region['y'] >= hiragana['y']:
        errors.append("Region must be strictly above hiragana")
    if classification['y'] >= number['y']:
        errors.append("Classification must be strictly above number")
    
    # Double check diagonals
    if region['x'] >= number['x'] or region['y'] >= number['y']:
        errors.append("Region must be strictly top-left of number")
    if classification['x'] <= hiragana['x'] or classification['y'] >= hiragana['y']:
        errors.append("Classification must be strictly top-right of hiragana")
    
    return len(errors) == 0, "; ".join(errors) if errors else "Valid positions"

def filter_valid_detections(sections):
    """Filter out invalid detections and keep only those in correct relative positions."""
    if not sections:
        return None
        
    # Group sections by class number
    section_groups = {i: [] for i in range(4)}  # Initialize all section types
    for section in sections:
        cls_number = section[0]
        if cls_number in section_groups:
            # Calculate center point
            x1, y1, x2, y2 = section[1:5]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            section_groups[cls_number].append((center_x, center_y, section))

    # Check if we have at least one of each section
    if any(len(group) == 0 for group in section_groups.values()):
        return None
    # Find best combination of sections that satisfy relative positions
    best_sections = []
    min_error = float('inf')

    # Try each possible combination
    for region in section_groups[0]:      # top-left
        for hiragana in section_groups[1]: # bottom-left
            for classif in section_groups[2]: # top-right
                for number in section_groups[3]: # bottom-right
                    rx, ry = region[0], region[1]
                    hx, hy = hiragana[0], hiragana[1]
                    cx, cy = classif[0], classif[1]
                    nx, ny = number[0], number[1]

                    # Check relative positions
                    if (rx < cx and rx < nx and  # region is leftmost
                        hx < nx and              # hiragana is left of number
                        ry < hy and              # region is above hiragana
                        cy < ny and              # classification is above number
                        rx < cx and              # region is left of classification
                        hx < nx):                # hiragana is left of number

                        # Calculate position error (smaller is better)
                        error = abs(rx - hx) + abs(cx - nx)  # Should be minimal for vertical alignment
                        if error < min_error:
                            min_error = error
                            best_sections = [
                                region[2],    # Original section tuple
                                hiragana[2],
                                classif[2],
                                number[2]
                            ]

    return best_sections if best_sections else None

def correct_section_positions(sections):
    """Attempt to correct section positions by sorting based on expected positions."""
    if not sections or len(set(s[0] for s in sections)) != 4:
        return None
        
    corrected_sections = []
    sections_by_type = {}
    
    # Group sections by type
    for section in sections:
        cls_number = section[0]
        sections_by_type[cls_number] = section
    
    # Sort sections into correct positions
    try:
        # region(0) should be top-left
        # classification(2) should be top-right
        # hiragana(1) should be bottom-left
        # number(3) should be bottom-right
        sorted_sections = [
            sections_by_type[0],  # region
            sections_by_type[2],  # classification
            sections_by_type[1],  # hiragana
            sections_by_type[3],  # number
        ]
        return sorted_sections
    except KeyError:
        return None

async def evaluate_detection():
    # Load models
    model_LicensePlateDet = YOLO(os.path.join(YOLO_DIR, "license_plate_detection/epoch30.pt"))
    model_sections = YOLO(os.path.join(YOLO_DIR, "section_detection/sections.pt"))
    
    # Load dataset
    eval_dataset_path = 'recognition-api/eval_dataset'
    images = load_eval_dataset(eval_dataset_path)
    
    total_images = len(images)
    license_plate_detected = 0
    section_detection_failed = []
    section_detection_success = 0
    
    print(f"Evaluating {total_images} images...")
    
    for image_path in images:
        print(f"\nProcessing: {image_path}")
        image = cv2.imread(image_path)
        filename = os.path.basename(image_path)
        
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Check license plate detection
        detections, image_lp = detect_license_plate(model_LicensePlateDet, image)
        
        if not detections:
            print(f"No license plate detected in: {image_path}")
            save_annotated_image(image, [], None, f"no_plate_{filename}")
            continue
            
        license_plate_detected += 1
        
        # For each detected license plate, check section detection
        for x1, y1, x2, y2 in detections:
            cropped = image_lp[y1:y2, x1:x2]
            sections, _ = detect_sections(model_sections, cropped)
            
            # Save original detection first
            save_annotated_image(image, detections, sections, filename)
            
            # Filter and validate sections
            valid_sections = filter_valid_detections(sections)
            
            if valid_sections is None and sections and len(set(s[0] for s in sections)) == 4:
                # Try to correct positions if we have all section types
                corrected_sections = correct_section_positions(sections)
                if corrected_sections:
                    # Save corrected positions with different color scheme
                    save_annotated_image(image, detections, corrected_sections, f"corrected_{filename}", is_filtered=True)
                
            if valid_sections is None:
                failure_info = {
                    'image': image_path,
                    'sections_found': len(sections) if sections else 0,
                }
                
                if sections:
                    section_types = {cls_[s[0]]: 1 for s in sections}
                    missing_sections = [s for s in cls_.values() if s not in section_types]
                    failure_info.update({
                        'detected_sections': list(section_types.keys()),
                        'missing_sections': missing_sections
                    })
                    
                    # Check positions if we have all sections
                    if len(section_types) == 4:
                        _, position_errors = validate_section_positions(sections)
                        failure_info['position_errors'] = position_errors
                
                section_detection_failed.append(failure_info)
                save_annotated_image(image, detections, sections, f"failed_{filename}")
                continue
            
            # Save filtered detection
            save_annotated_image(image, detections, valid_sections, filename, is_filtered=True)
            section_detection_success += 1
    
    # Calculate accuracies
    license_plate_accuracy = (license_plate_detected / total_images) * 100
    section_accuracy = (section_detection_success / license_plate_detected) * 100 if license_plate_detected > 0 else 0
    
    # Print results
    print("\n=== Detection Evaluation Results ===")
    print(f"Total images evaluated: {total_images}")
    print(f"\nLicense Plate Detection:")
    print(f"Success: {license_plate_detected}/{total_images}")
    print(f"Accuracy: {license_plate_accuracy:.2f}%")
    
    print(f"\nSection Detection:")
    print(f"Success: {section_detection_success}/{license_plate_detected}")
    print(f"Accuracy: {section_accuracy:.2f}%")
    
    if section_detection_failed:
        print("\nFailed Section Detections:")
        for failure in section_detection_failed:
            print(f"\nImage: {failure['image']}")
            print(f"Sections found: {failure['sections_found']}")
            if 'detected_sections' in failure:
                print(f"Detected sections: {', '.join(failure['detected_sections'])}")
            if 'position_errors' in failure:
                print(f"Position errors: {failure['position_errors']}")

cls_ = {
    0: "region",
    1: "hiragana",
    2: "classification",
    3: "number"
}

if __name__ == "__main__":
    asyncio.run(evaluate_detection())

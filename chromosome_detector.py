import cv2
import numpy as np
import os
from pathlib import Path


def extract_chromosomes(image_path,output_dir,min_area,max_area,padding):
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Directory ready: {os.path.abspath(output_dir)}/")
    
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    print(f"Image loaded successfully")
    print(f"  - Image size: {img.shape[1]} x {img.shape[0]} pixels")
    print(f"  - Color channels: {img.shape[2]}")

    
    # Convert to grayscale(openCV function)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale conversion complete")

    
    # Apply Gaussian blur to reduce noise
    # kernel size: 5x5
    # 0 (automatically select used SD value)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print(f"Noise reduction complete")

    
    # Apply thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"Binary segmentation complete")
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    print(f"Image cleanup complete")

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")


    # Filter contours by area
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            valid_contours.append(cnt)
    
    print(f"Filtered to {len(valid_contours)} valid chromosomes")


    # draw the chromosome boundaries box
    img_with_boxes = img.copy()

    # Store metadata (id, position, size, area, perimeter) of extracted chromosomes.
    chromosome_info = [] 
    for idx, cnt in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Calculate area and perimeter
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Add padding around the chromosome
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        # Extract chromosome region
        chromosome = img[y_start:y_end, x_start:x_end]
        
        # Save extracted chromosome
        output_path = os.path.join(output_dir, f'chromosome_{idx+1:03d}.png')
        cv2.imwrite(output_path, chromosome)
        
        # Store info
        chromosome_info.append({
            'id': idx + 1,
            'position': (x, y),
            'size': (w, h),
            'area': area,
            'perimeter': perimeter
        })

        # Prints save status, size, and area in the console.
        print(f"Saved: chromosome_{idx+1:03d}.png ({w}x{h} pixels, area={int(area)})")
        
        # Draw rectangle and label on visualization image
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img_with_boxes, str(idx+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        

    # Save annotated image
    annotated_path = os.path.join(output_dir, 'annotated_image.png')

    # img_with_boxes → chromosome boundaries + labels draw image
    cv2.imwrite(annotated_path, img_with_boxes)
    print(f"Saved: annotated_image.png")

    # Save binary mask
    binary_path = os.path.join(output_dir, 'binary_mask.png')
    cv2.imwrite(binary_path, binary)
    print(f"Saved: binary_mask.png")


    # Final summary

    print(f"\n Output location: {os.path.abspath(output_dir)}/")
    print(f"\n Summary:")
    print(f"  - Total chromosomes detected: {len(valid_contours)}")
    return len(valid_contours)


def main():
 
    try:

        image_path="input/chromosome_image.png"
        output_dir='output'
        # Check if input file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        # Run detection
        num_chromosomes = extract_chromosomes(
            image_path="input/chromosome_image.png",
            output_dir='output',
            min_area=10,
            max_area=50000,
            padding=20
        )
        
        print(f"SUCCESS: Extracted {num_chromosomes} chromosomes!")
        print(f"\nAll results saved to: {os.path.abspath(output_dir)}/")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please make sure the input image exists in the 'input' folder.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        print("Please check your input image and try again.")


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import os
from pathlib import Path


def extract_chromosomes(image_path,output_dir,min_area,max_area,padding):
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Directory ready: {os.path.abspath(output_dir)}/")
    
    # Read image
    # image load function of openCV 
    img = cv2.imread(image_path)
    # load image as numpy array.
    # if image grayscale, store as 2D array
    # if image is color , store as 3D array
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    print(f"Image loaded successfully")
    # img.shape[0] → Height(ROW)
    #img.shape[1] → Width(COLUMNS)
    print(f"  - Image size: {img.shape[1]} x {img.shape[0]} pixels")
    print(f"  - Color channels: {img.shape[2]}")


    
    # Convert to grayscale(openCV function)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 0 → black
    # 255 → white
    # between 0 , 255 → grey
    print(f"Grayscale conversion complete")

    
    # Apply Gaussian blur to reduce noise
    # kernel size: 5x5
    # Standard deviation — 0 (automatically select used SD value)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    print(f"Noise reduction complete")


    
    # Apply thresholding
    # Thresholding mean convert grayscale image to black and white(binary) image
    # best threshold value automatically calculated(Otsu method automatically decide)
    # Pixel > threshold → black (0)  (done by cv2.THRESH_BINARY_INV)
    # Pixel ≤ threshold → white (255) (done by cv2.THRESH_BINARY_INV)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(f"Binary segmentation complete")


    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    # (Closing operation) Small holes / cracks fill (Chromosome shapes smooth)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # (Opening operation) Small noise / isolated pixels remove (Main chromosome shapes preserve)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    print(f"Image cleanup complete")


    
    # Find contours
    # Contour = boundary / shape outline of object
    # Contour is the line that traces the outer outline/shape of the white object in the image.
    # Contour = A collection of points that trace the edges of a chromosome. 
    # These points can be used to crop, measure, and analyze objects.
    # cv2.RETR_EXTERNAL → External contours only (outer boundary)
    # cv2.CHAIN_APPROX_SIMPLE → Contour approximation
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Example: Found 12 contours
    # mean → 12 chromosomes / objects were detected in the binary image

    print(f"Found {len(contours)} contours")



    # Filter contours by area
    valid_contours = []
    for cnt in contours:
        # The cv2.contourArea() function calculates the area (pixel count) of the contour.
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            valid_contours.append(cnt)
    
    print(f"Filtered to {len(valid_contours)} valid chromosomes")



    # Extract and save individual chromosomes
    # draw the chromosome boundaries box
    img_with_boxes = img.copy()

    # Store metadata (id, position, size, area, perimeter) of extracted chromosomes.
    chromosome_info = [] 
    for idx, cnt in enumerate(valid_contours):
        
        # Get bounding rectangle
        # (x, y) → rectangle top-left corner
        # (w, h) → width and height
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
        # Cut out the chromosome area from the original image.
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
        # cv2.rectangle → green rectangle draw (chromosome boundary)
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # cv2.putText → chromosome index label display
        cv2.putText(img_with_boxes, str(idx+1), (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Save annotated image
    # os.path.join → output folder + file name combine function
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
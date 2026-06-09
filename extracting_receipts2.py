import cv2
import numpy as np
import argparse
import os
import sys
from typing import List, Tuple

def get_reading_order_sorted_contours(contours: List[np.ndarray], img_width: int, img_height: int) -> List[np.ndarray]:
    """
    Sorts contours in reading order: top-to-bottom, then left-to-right.
    Uses row clustering based on vertical center proximity.
    """
    if not contours:
        return []
    
    # Get bounding boxes and centers
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cy = y + h / 2
        boxes.append((cy, x, cnt))
    
    # Sort by Y center
    boxes.sort(key=lambda b: b[0])
    
    # Group into rows (tolerance = 5% of image height)
    row_tolerance = img_height * 0.05
    rows = []
    current_row = [boxes[0]]
    
    for i in range(1, len(boxes)):
        if abs(boxes[i][0] - current_row[0][0]) < row_tolerance:
            current_row.append(boxes[i])
        else:
            # Sort current row by X and add to rows
            current_row.sort(key=lambda b: b[1])
            rows.extend([b[2] for b in current_row])
            current_row = [boxes[i]]
    
    # Add last row
    current_row.sort(key=lambda b: b[1])
    rows.extend([b[2] for b in current_row])
    
    return rows

def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Applies perspective transform."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def find_best_4_vertices(cnt: np.ndarray) -> np.ndarray:
    """Finds exact 4 corners or falls back to minAreaRect."""
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    # Try epsilon factors from tight to loose
    for eps_factor in [0.01, 0.02, 0.03, 0.05, 0.07]:
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    # Fallback to minAreaRect
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return np.array(box, dtype="float32")

def merge_nearby_boxes(boxes: List[Tuple[int, int, int, int]], threshold_px: int = 20) -> List[Tuple[int, int, int, int]]:
    """
    Conservatively merges boxes only if they are very close (within threshold_px).
    This joins fragmented parts of the same receipt without fusing separate receipts.
    """
    if not boxes:
        return []
    
    # Sort by Y then X for consistent processing
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
            
        x, y, w, h = boxes[i]
        curr_box = [x, y, w, h]
        used[i] = True
        
        # Iteratively merge close boxes
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                
                x2, y2, w2, h2 = boxes[j]
                
                # Calculate gap between boxes
                gap_x = max(0, x2 - (curr_box[0] + curr_box[2]), curr_box[0] - (x2 + w2))
                gap_y = max(0, y2 - (curr_box[1] + curr_box[3]), curr_box[1] - (y2 + h2))
                
                # Only merge if gap is small AND there's substantial overlap in at least one dimension
                if gap_x < threshold_px and gap_y < threshold_px:
                    # Calculate overlap in each dimension
                    overlap_x = min(curr_box[0] + curr_box[2], x2 + w2) - max(curr_box[0], x2)
                    overlap_y = min(curr_box[1] + curr_box[3], y2 + h2) - max(curr_box[1], y2)
                    
                    min_width = min(curr_box[2], w2)
                    min_height = min(curr_box[3], h2)
                    
                    # Require at least 70% overlap in one dimension
                    if overlap_x > 0.7 * min_width or overlap_y > 0.7 * min_height:
                        # Merge
                        new_x = min(curr_box[0], x2)
                        new_y = min(curr_box[1], y2)
                        new_w = max(curr_box[0] + curr_box[2], x2 + w2) - new_x
                        new_h = max(curr_box[1] + curr_box[3], y2 + h2) - new_y
                        curr_box = [new_x, new_y, new_w, new_h]
                        used[j] = True
                        changed = True
        
        merged.append(tuple(curr_box))
    
    return merged

def split_tall_boxes(boxes: List[Tuple[int, int, int, int]], mask: np.ndarray, aspect_threshold: float = 1.5) -> List[Tuple[int, int, int, int]]:
    """
    Splits tall boxes that likely contain multiple stacked receipts.
    Looks for the best split point near the middle of the box.
    """
    from scipy.ndimage import gaussian_filter1d
    result = []
    
    for x, y, w, h in boxes:
        aspect = h / w if w > 0 else 0
        
        # If not too tall, keep as is
        if aspect < aspect_threshold:
            result.append((x, y, w, h))
            continue
        
        print(f"DEBUG: Analyzing tall box ({x},{y},{w},{h}) aspect={aspect:.2f}")
        
        # Extract region and analyze horizontal projection
        roi = mask[y:y+h, x:x+w]
        projection = np.sum(roi, axis=1) / 255  # Count white pixels per row
        
        # Smooth projection
        smoothed = gaussian_filter1d(projection, sigma=5)
        
        # Look for the best split point in the middle 40% of the box
        mid_start = int(h * 0.3)
        mid_end = int(h * 0.7)
        mid_region = smoothed[mid_start:mid_end]
        
        if len(mid_region) > 0:
            # Find minimum in middle region
            local_min_idx = np.argmin(mid_region)
            split_point = mid_start + local_min_idx
            
            # Check if this is a significant valley
            if smoothed[split_point] < np.mean(smoothed) * 0.4:
                # Split here
                result.append((x, y, w, split_point))
                result.append((x, y + split_point, w, h - split_point))
                print(f"DEBUG: Split into 2 parts at y={split_point}")
            else:
                # No clear valley, keep as is
                result.append((x, y, w, h))
                print(f"DEBUG: No clear split point found, keeping as is")
        else:
            result.append((x, y, w, h))
    
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract multiple receipts from an image.")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Directory to save extracted receipts")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not read image: {args.input}")
        sys.exit(1)

    orig_height, orig_width = image.shape[:2]
    print(f"Processing image: {orig_width}x{orig_height}")
    
    # 1. Scaling for detection (consistent processing size)
    det_width = 1600
    ratio = orig_width / det_width
    det_height = int(orig_height / ratio)
    resized = cv2.resize(image, (det_width, det_height))
    
    # 2. Preprocessing - Try both tight and loose morphology
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=25,
        C=12
    )
    
    # Canny edges
    edges = cv2.Canny(blurred, 50, 150)
    combined = cv2.bitwise_or(thresh, edges)
    
    # Remove tiny noise
    kernel_tiny = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_tiny)
    
    # TIGHT morphology (separates receipts better but may fragment them)
    kernel_tight = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_tight = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_tight, iterations=1)
    
    # Dilate to expand regions (for receipts with sparse text like Denner)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_tight_dilated = cv2.dilate(closed_tight, kernel_dilate, iterations=2)
    
    # LOOSE morphology (connects text better but may merge receipts)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    closed_h = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_h, iterations=2)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    closed_loose = cv2.morphologyEx(closed_h, cv2.MORPH_CLOSE, kernel_v, iterations=2)
    
    # Try both and use whichever gives better results
    closed = closed_tight  # Start with tight
    
    if args.debug:
        cv2.imwrite(os.path.join(args.output, "debug_mask_tight.jpg"), closed_tight)
        cv2.imwrite(os.path.join(args.output, "debug_mask_tight_dilated.jpg"), closed_tight_dilated)
        cv2.imwrite(os.path.join(args.output, "debug_mask_loose.jpg"), closed_loose)
    
    # 3. Try tight, tight+dilated, and loose morphology
    det_area = det_width * det_height
    
    def count_valid_candidates(mask_input):
        cnts, _ = cv2.findContours(mask_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        max_aspect = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if not (det_area * 0.008 < area < det_area * 0.6):
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 6:
                continue
            count += 1
            max_aspect = max(max_aspect, h / w if w > 0 else 0)
        return count, cnts, max_aspect
    
    tight_count, tight_cnts, tight_max_aspect = count_valid_candidates(closed_tight)
    tight_dil_count, tight_dil_cnts, tight_dil_max_aspect = count_valid_candidates(closed_tight_dilated)
    loose_count, loose_cnts, loose_max_aspect = count_valid_candidates(closed_loose)
    
    print(f"DEBUG: Tight: {tight_count} (aspect={tight_max_aspect:.2f}), Tight+Dilated: {tight_dil_count} (aspect={tight_dil_max_aspect:.2f}), Loose: {loose_count} (aspect={loose_max_aspect:.2f})")
    
    # Use best option: prefer tight morphology if it gives 2-6 candidates
    if 2 <= tight_count <= 6:
        closed = closed_tight
        cnts = tight_cnts
        print("DEBUG: Using TIGHT morphology")
    elif 2 <= tight_dil_count <= 6:
        closed = closed_tight_dilated
        cnts = tight_dil_cnts
        print("DEBUG: Using TIGHT+DILATED morphology")
    elif 2 <= loose_count <= 6:
        closed = closed_loose
        cnts = loose_cnts
        print("DEBUG: Using LOOSE morphology")
    else:
        # Default to tight
        closed = closed_tight
        cnts = tight_cnts
        print("DEBUG: Using TIGHT morphology (default)")
    
    if args.debug:
        cv2.imwrite(os.path.join(args.output, "debug_mask_final.jpg"), closed)
    
    # 3. Contour Detection
    print(f"DEBUG: Found {len(cnts)} raw contours")
    candidate_boxes = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        # Area filtering: between 0.8% and 60% of detection image
        if not (det_area * 0.008 < area < det_area * 0.6):
            continue
            
        # Aspect ratio filtering: Receipts are roughly rectangular
        # Allow 1:6 to 6:1 (handles both portrait and landscape)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 6:
            continue
            
        candidate_boxes.append((x, y, w, h, c))
        print(f"DEBUG: Candidate box: ({x},{y},{w},{h}), area: {area:.0f}, aspect: {aspect:.2f}")
    
    print(f"DEBUG: {len(candidate_boxes)} candidates after filtering")
    
    # Skip merging - rely on splitting instead to separate stacked receipts
    merged_boxes = [(b[0], b[1], b[2], b[3]) for b in candidate_boxes]
    
    print(f"DEBUG: {len(merged_boxes)} boxes (no merging)")
    
    # Split tall boxes that likely contain multiple stacked receipts
    split_boxes = split_tall_boxes(merged_boxes, closed, aspect_threshold=1.5)
    
    print(f"DEBUG: {len(split_boxes)} boxes after splitting")
    
    if not split_boxes:
        print("Error: No receipt regions detected.")
        sys.exit(1)
    
    # Convert back to original resolution contours
    final_regions = []
    for x, y, w, h in split_boxes:
        # Create a rectangular contour from the bounding box
        pts = np.array([
            [[x, y]], 
            [[x + w, y]], 
            [[x + w, y + h]], 
            [[x, y + h]]
        ])
        orig_pts = (pts * ratio).astype(np.int32)
        final_regions.append(orig_pts)

    # Sort in reading order
    sorted_cnts = get_reading_order_sorted_contours(final_regions, orig_width, orig_height)
    print(f"DEBUG: Processing {len(sorted_cnts)} final regions")

    # 4. Extraction
    count = 0
    debug_img = image.copy() if args.debug else None
    
    for i, cnt in enumerate(sorted_cnts):
        receipt_id = f"{i+1:03d}"
        
        try:
            vertices = find_best_4_vertices(cnt)
            unwarped = four_point_transform(image, vertices)
            
            # Filter out extreme aspect ratios (likely errors)
            h_crop, w_crop = unwarped.shape[:2]
            aspect = max(h_crop, w_crop) / max(min(h_crop, w_crop), 1)
            if aspect > 8:  # Skip if extremely long and thin
                print(f"DEBUG: Skipping receipt {receipt_id} - extreme aspect ratio {aspect:.1f}")
                continue
            
            output_filename = os.path.join(args.output, f"receipt_{receipt_id}.jpg")
            cv2.imwrite(output_filename, unwarped)
            count += 1
            print(f"Saved: receipt_{receipt_id}.jpg ({w_crop}x{h_crop})")
            
            if args.debug:
                pts = vertices.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug_img, [pts], True, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.putText(debug_img, f"#{receipt_id}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        except Exception as e:
            print(f"DEBUG: Error processing contour {i}: {e}")
            continue

    if args.debug:
        cv2.imwrite(os.path.join(args.output, "debug_boxes.jpg"), debug_img)
    
    print(f"Extraction complete. Found {count} receipts.")

if __name__ == "__main__":
    main()
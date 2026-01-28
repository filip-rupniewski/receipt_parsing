import cv2
import numpy as np
import argparse
import os
import sys
from typing import List, Tuple, Optional

def get_reading_order_sorted_contours(contours: List[np.ndarray], img_width: int, img_height: int) -> List[np.ndarray]:
    """
    Sorts contours in reading order: top-to-bottom, then left-to-right.
    Uses a tighter row-based grouping approach for dense layouts.
    """
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h, cnt))
    
    if not boxes:
        return []

    # Sort primarily by center Y
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    
    rows = []
    if boxes:
        current_row = [boxes[0]]
        for i in range(1, len(boxes)):
            prev_y_center = current_row[0][1] + current_row[0][3] / 2
            curr_y_center = boxes[i][1] + boxes[i][3] / 2
            # Tighter threshold for dense receipts (5% of image height)
            if abs(curr_y_center - prev_y_center) < (img_height * 0.05):
                current_row.append(boxes[i])
            else:
                current_row.sort(key=lambda b: b[0])
                rows.extend(current_row)
                current_row = [boxes[i]]
        current_row.sort(key=lambda b: b[0])
        rows.extend(current_row)
        
    return [r[4] for r in rows]

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
    for eps_factor in np.linspace(0.01, 0.05, 10):
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    # Fallback to minAreaRect
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return np.array(box, dtype="float32")

def merge_nearby_boxes(boxes: List[Tuple[int, int, int, int]], threshold_px: int) -> List[Tuple[int, int, int, int]]:
    """Merges boxes that are close to each other, with vertical affinity for aligned boxes."""
    if not boxes:
        return []
    
    # Sort boxes (Y then X)
    boxes.sort(key=lambda b: (b[1], b[0]))
    
    merged = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
        
        curr_box = list(boxes[i])
        used[i] = True
        
        while True:
            added = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                
                x1, y1, w1, h1 = curr_box
                x2, y2, w2, h2 = boxes[j]
                
                # Calculate gaps
                dx = max(0, x1 - (x2 + w2), x2 - (x1 + w1))
                dy = max(0, y1 - (y2 + h2), y2 - (y1 + h1))
                
                # Check for horizontal alignment (overlap in X range)
                x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
                is_aligned_v = x_overlap > (min(w1, w2) * 0.6)
                
                # Vertical affinity: Allow much larger vertical gap if horizontally aligned
                # (to join logo/header to body)
                v_threshold = threshold_px * 5 if is_aligned_v else threshold_px
                
                is_nearby = (dx < threshold_px and dy < v_threshold)
                
                if is_nearby:
                    # Merge boxes
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    curr_box = [nx, ny, nw, nh]
                    used[j] = True
                    added = True
            if not added:
                break
        merged.append(tuple(curr_box))
        
    return merged

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract multiple receipts from an image.")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Directory to save extracted receipts")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    image = cv2.imread(args.input)
    if image is None:
        sys.exit(1)

    orig_height, orig_width = image.shape[:2]
    
    # 1. Scaling for detection (consistent kernel sizes)
    det_width = 1600
    ratio = orig_width / det_width
    det_height = int(orig_height / ratio)
    resized = cv2.resize(image, (det_width, det_height))
    
    # 2. Dual-Path Preprocessing
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Path A: Adaptive Threshold for Text/Logos
    blurred_t = cv2.GaussianBlur(gray, (5, 5), 0)
    # Reduced C (10 -> 5) to include more pixels as foreground
    thresh = cv2.adaptiveThreshold(blurred_t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 51, 5)
    
    # Path B: Canny for Shape Boundaries
    blurred_e = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred_e, 30, 100)
    
    # Combine (Logical OR)
    combined = cv2.bitwise_or(thresh, edges)
    
    # Balanced Morphology: Join pieces within a receipt without bridging gaps between them
    # Join horizontal gaps first (text lines)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated_h = cv2.dilate(combined, h_kernel, iterations=1)
    
    # Join vertical gaps (body parts)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    joined = cv2.dilate(dilated_h, v_kernel, iterations=1)
    
    # Closing to solidify
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Contour Detection
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_boxes = []
    # Area threshold: 0.2% of detection area (enough for small receipts in a grid)
    min_area = (det_width * det_height) * 0.002
    
    for c in cnts:
        if cv2.contourArea(c) > min_area:
            candidate_boxes.append(cv2.boundingRect(c))
    
    # Tighter merge threshold to avoid connecting independent receipts (1% of width)
    merge_thresh = int(det_width * 0.01)
    merged_boxes = merge_nearby_boxes(candidate_boxes, merge_thresh)
    
    debug_img = image.copy()
    if not merged_boxes:
        cv2.imwrite(os.path.join(args.output, "debug_boxes.jpg"), debug_img)
        print("Error: No receipt regions detected.")
        sys.exit(1)

    # Convert to original resolution and creates contours
    final_regions = []
    for x, y, w, h in merged_boxes:
        pts = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]])
        orig_pts = (pts * ratio).astype(np.int32)
        final_regions.append(orig_pts)

    # Sort in reading order (tight grouping for grids)
    sorted_cnts = get_reading_order_sorted_contours(final_regions, orig_width, orig_height)

    # 4. Extraction
    count = 0
    for i, cnt in enumerate(sorted_cnts):
        receipt_id = f"{i+1:03d}"
        output_filename = os.path.join(args.output, f"receipt_{receipt_id}.jpg")
        
        # Vertex Refinement: Try to find real vertices within the detected bounding box region
        # We perform a local refinement to handle minor perspective/rotation better
        vertices = find_best_4_vertices(cnt)
        
        try:
            unwarped = four_point_transform(image, vertices)
            # Filter out extreme aspect ratio crops (likely noise or split failure)
            h_crop, w_crop = unwarped.shape[:2]
            aspect = h_crop / w_crop if w_crop > 0 else 0
            if aspect > 10 or aspect < 0.1:
                continue
                
            cv2.imwrite(output_filename, unwarped)
            count += 1
        except Exception:
            continue
        
        # Debug drawing
        pts = vertices.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 10)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(debug_img, f"#{receipt_id}", (x, y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 10)

    cv2.imwrite(os.path.join(args.output, "debug_boxes.jpg"), debug_img)
    print(f"Extraction complete. Found {count} receipts.")

if __name__ == "__main__":
    main()

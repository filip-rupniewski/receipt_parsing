import cv2
import numpy as np
import argparse
import os
import sys
from typing import List, Tuple
from scipy import ndimage

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xA = max(x1, x2); yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2); yB = min(y1 + h1, y2 + h2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea / float(w1 * h1 + w2 * h2 - interArea + 1e-6)

def sort_receipts_reading_order(boxes: List[Tuple], img_height: int) -> List[Tuple]:
    if not boxes: return []
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    sorted_boxes, current_row = [], [boxes[0]]
    row_tol = img_height * 0.15
    for i in range(1, len(boxes)):
        if abs((boxes[i][1] + boxes[i][3]/2) - (current_row[0][1] + current_row[0][3]/2)) < row_tol:
            current_row.append(boxes[i])
        else:
            current_row.sort(key=lambda b: b[0])
            sorted_boxes.extend(current_row)
            current_row = [boxes[i]]
    current_row.sort(key=lambda b: b[0])
    sorted_boxes.extend(current_row)
    return sorted_boxes

def split_merged_contour_by_projection(contour: np.ndarray, mask: np.ndarray, 
                                       min_gap_ratio: float = 0.02) -> List[np.ndarray]:
    """
    Rozdziela zlany kontur na podstawie analizy projekcji.
    Szuka wyraźnych przerw (pustych przestrzeni) między paragonami.
    """
    x, y, w, h = cv2.boundingRect(contour)
    
    # Wyciągnij maskę dla tego konturu
    roi_mask = mask[y:y+h, x:x+w]
    
    # Sprawdź czy kontur jest poziomy czy pionowy
    aspect = max(w, h) / min(w, h)
    
    if aspect > 2.0 and h > w:
        # Pionowy - analizuj projekcję poziomą
        projection = np.sum(roi_mask, axis=1) / 255.0
        projection_smooth = ndimage.gaussian_filter1d(projection, sigma=10)
        
        # Znajdź minima (przerwy) w środkowej 60% projekcji
        mid_start = int(len(projection_smooth) * 0.2)
        mid_end = int(len(projection_smooth) * 0.8)
        mid_proj = projection_smooth[mid_start:mid_end]
        
        # Próg dla przerwy - mniej niż 10% maksymalnej wartości
        threshold = np.max(projection_smooth) * 0.15
        
        split_points = []
        for i in range(1, len(mid_proj) - 1):
            if mid_proj[i] < threshold and mid_proj[i] < mid_proj[i-1] and mid_proj[i] < mid_proj[i+1]:
                split_points.append(mid_start + i)
        
        if not split_points:
            return [contour]
        
        # Podziel kontur w znalezionych punktach
        result_contours = []
        prev_y = 0
        for split_y in sorted(split_points):
            if split_y - prev_y > h * min_gap_ratio:
                # Utwórz prostokątny kontur dla tej sekcji
                section_h = split_y - prev_y
                pts = np.array([[x, y + prev_y], [x + w, y + prev_y], 
                               [x + w, y + split_y], [x, y + split_y]], dtype=np.int32)
                result_contours.append(pts)
                prev_y = split_y
        
        # Ostatnia sekcja
        if h - prev_y > h * min_gap_ratio:
            pts = np.array([[x, y + prev_y], [x + w, y + prev_y], 
                           [x + w, y + h], [x, y + h]], dtype=np.int32)
            result_contours.append(pts)
        
        return result_contours if len(result_contours) > 1 else [contour]
    
    elif aspect > 2.0 and w > h:
        # Poziomy - analizuj projekcję pionową
        projection = np.sum(roi_mask, axis=0) / 255.0
        projection_smooth = ndimage.gaussian_filter1d(projection, sigma=10)
        
        mid_start = int(len(projection_smooth) * 0.2)
        mid_end = int(len(projection_smooth) * 0.8)
        mid_proj = projection_smooth[mid_start:mid_end]
        
        threshold = np.max(projection_smooth) * 0.15
        
        split_points = []
        for i in range(1, len(mid_proj) - 1):
            if mid_proj[i] < threshold and mid_proj[i] < mid_proj[i-1] and mid_proj[i] < mid_proj[i+1]:
                split_points.append(mid_start + i)
        
        if not split_points:
            return [contour]
        
        result_contours = []
        prev_x = 0
        for split_x in sorted(split_points):
            if split_x - prev_x > w * min_gap_ratio:
                section_w = split_x - prev_x
                pts = np.array([[x + prev_x, y], [x + split_x, y], 
                               [x + split_x, y + h], [x + prev_x, y + h]], dtype=np.int32)
                result_contours.append(pts)
                prev_x = split_x
        
        if w - prev_x > w * min_gap_ratio:
            pts = np.array([[x + prev_x, y], [x + w, y], 
                           [x + w, y + h], [x + prev_x, y + h]], dtype=np.int32)
            result_contours.append(pts)
        
        return result_contours if len(result_contours) > 1 else [contour]
    
    return [contour]

def detect_with_aggressive_morphology(gray: np.ndarray, output_dir: str, debug: bool) -> Tuple[List, np.ndarray]:
    """Agresywna morfologia + Convex Hull"""
    print("\n🔍 Metoda: Agresywna morfologia + Convex Hull")
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize=51, C=15)
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, "debug_thresh.jpg"), thresh)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=4)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=3)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, "debug_morph_aggressive.jpg"), cleaned)
    
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Znaleziono {len(cnts)} konturów")
    
    return cnts, cleaned

def detect_with_paper_color(image: np.ndarray, gray: np.ndarray, output_dir: str, debug: bool) -> Tuple[List, np.ndarray]:
    """Detekcja koloru papieru w HSV"""
    print("\n🔍 Metoda: Detekcja koloru papieru (HSV)")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_paper = np.array([0, 0, 180])
    upper_paper = np.array([180, 50, 255])
    
    mask = cv2.inRange(hsv, lower_paper, upper_paper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, "debug_paper_color.jpg"), mask)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"   Znaleziono {len(cnts)} konturów")
    
    return cnts, mask

def process_contours(cnts: np.ndarray, mask: np.ndarray, det_area: int, image_shape: Tuple[int, int], 
                     output_dir: str, debug: bool, method_name: str) -> List:
    """Przetwarza kontury i zwraca listę zaakceptowanych paragonów"""
    
    accepted = []
    det_w, det_h = image_shape
    
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area < det_area * 0.03 or area > det_area * 0.75:
            continue
        
        # Spróbuj rozdzielić zlany kontur
        split_contours = split_merged_contour_by_projection(c, mask, min_gap_ratio=0.03)
        
        for split_c in split_contours:
            hull = cv2.convexHull(split_c)
            rect = cv2.minAreaRect(hull)
            (cx, cy), (rw, rh), angle = rect
            
            if rw == 0 or rh == 0:
                continue
            
            hull_area = cv2.contourArea(hull)
            rect_area = rw * rh
            solidity = hull_area / float(rect_area) if rect_area > 0 else 0
            
            if solidity < 0.60:
                continue
            
            aspect = max(rw, rh) / min(rw, rh)
            if aspect < 1.3 or aspect > 6.5:
                continue
            
            box = cv2.boxPoints(rect)
            x, y, w, h = cv2.boundingRect(box.astype(np.int32))
            curr_bbox = (x, y, w, h)
            
            is_dup = False
            for acc in accepted:
                if calculate_iou(curr_bbox, (acc[0], acc[1], acc[2], acc[3])) > 0.3:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            print(f"   ✓ Paragon {len(accepted)+1}: area={area:.0f}, aspect={aspect:.2f}, solidity={solidity:.2f}")
            accepted.append((*curr_bbox, box))
    
    print(f"   Metoda '{method_name}' znalazła {len(accepted)} paragonów")
    return accepted

def main() -> None:
    parser = argparse.ArgumentParser(description="Wykrywanie paragonów - pełna kartka")
    parser.add_argument("--input", required=True, help="Ścieżka do zdjęcia")
    parser.add_argument("--output", required=True, help="Katalog wyjściowy")
    parser.add_argument("--debug", action="store_true", help="Zapisz obrazy debugowania")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Błąd: Nie znaleziono pliku: {args.input}"); sys.exit(1)
    os.makedirs(args.output, exist_ok=True)
    
    image = cv2.imread(args.input)
    if image is None:
        print(f"Błąd: Nie można odczytać obrazu: {args.input}"); sys.exit(1)

    orig_h, orig_w = image.shape[:2]
    det_w = 1600
    ratio = orig_w / det_w
    det_h = int(orig_h / ratio)
    resized = cv2.resize(image, (det_w, det_h))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    det_area = det_w * det_h
    best_receipts = []
    best_method = ""
    
    # Metoda 1: Agresywna morfologia
    cnts1, mask1 = detect_with_aggressive_morphology(gray, args.output, args.debug)
    receipts1 = process_contours(cnts1, mask1, det_area, (det_w, det_h), args.output, args.debug, "morph")
    if 1 <= len(receipts1) <= 10 and len(receipts1) > len(best_receipts):
        best_receipts = receipts1
        best_method = "morph"
    
    # Metoda 2: Detekcja koloru papieru
    cnts2, mask2 = detect_with_paper_color(resized, gray, args.output, args.debug)
    receipts2 = process_contours(cnts2, mask2, det_area, (det_w, det_h), args.output, args.debug, "hsv")
    if 1 <= len(receipts2) <= 10 and len(receipts2) > len(best_receipts):
        best_receipts = receipts2
        best_method = "hsv"
    
    if not best_receipts:
        print("\n❌ Błąd: Nie wykryto paragonów.")
        print("   Sprawdź pliki debug_*.jpg w katalogu wyjściowym")
        sys.exit(1)
    
    print(f"\n✅ Wybrano metodę: {best_method} ({len(best_receipts)} paragonów)")
    
    final_receipts = []
    for x, y, w, h, box in best_receipts:
        orig_box = (box * ratio).astype(np.int32)
        orig_bbox = (int(x*ratio), int(y*ratio), int(w*ratio), int(h*ratio))
        final_receipts.append((*orig_bbox, orig_box))
    
    sorted_receipts = sort_receipts_reading_order(final_receipts, orig_h)
    
    count = 0
    debug_img = image.copy() if args.debug else None
    
    for i, (x, y, w, h, box_pts) in enumerate(sorted_receipts):
        rid = f"{i+1:03d}"
        try:
            unwarped = four_point_transform(image, box_pts)
            out_path = os.path.join(args.output, f"receipt_{rid}.jpg")
            cv2.imwrite(out_path, unwarped)
            count += 1
            print(f"💾 Zapisano: receipt_{rid}.jpg ({unwarped.shape[1]}x{unwarped.shape[0]})")
            if args.debug:
                pts = box_pts.reshape((-1, 1, 2))
                cv2.polylines(debug_img, [pts], True, (0, 255, 0), 4)
                cv2.putText(debug_img, f"#{rid}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        except Exception as e:
            print(f"⚠️  Błąd przy {rid}: {e}")
            
    if args.debug:
        cv2.imwrite(os.path.join(args.output, "debug_final.jpg"), debug_img)
        
    print(f"\n🎉 Gotowe! Wyodrębniono {count} paragonów do: {args.output}")

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import argparse
import os
import sys
from typing import List, Tuple

def order_points(pts: np.ndarray) -> np.ndarray:
    """Porządkuje 4 punkty w kolejności: TL (góra-lewo), TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Wykonuje transformację perspektywy (odwracanie rzutu)."""
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

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Oblicza Intersection over Union (IoU) dla dwóch bounding boxów [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = w1 * h1
    box2Area = w2 * h2

    # Dodajemy epsilon (1e-6) aby uniknąć dzielenia przez zero
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

def sort_receipts_reading_order(boxes: List[Tuple], img_height: int) -> List[Tuple]:
    """Sortuje paragony od góry do dołu i od lewej do prawej."""
    if not boxes: return []
    
    # Sortuj po środku Y (oś pionowa)
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    
    sorted_boxes = []
    current_row = [boxes[0]]
    # Tolerancja 10% wysokości obrazu dla "tego samego rzędu"
    row_tolerance = img_height * 0.10 
    
    for i in range(1, len(boxes)):
        if abs((boxes[i][1] + boxes[i][3]/2) - (current_row[0][1] + current_row[0][3]/2)) < row_tolerance:
            current_row.append(boxes[i])
        else:
            current_row.sort(key=lambda b: b[0]) # Sortuj w rzędzie od lewej do prawej
            sorted_boxes.extend(current_row)
            current_row = [boxes[i]]
            
    current_row.sort(key=lambda b: b[0])
    sorted_boxes.extend(current_row)
    
    return sorted_boxes

def main() -> None:
    parser = argparse.ArgumentParser(description="Wyodrębnij wiele paragonów z jednego zdjęcia.")
    parser.add_argument("--input", required=True, help="Ścieżka do zdjęcia wejściowego")
    parser.add_argument("--output", required=True, help="Katalog do zapisu paragonów")
    parser.add_argument("--debug", action="store_true", help="Zapisz obrazy debugowania")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Błąd: Nie znaleziono pliku: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    image = cv2.imread(args.input)
    if image is None:
        print(f"Błąd: Nie można odczytać obrazu: {args.input}")
        sys.exit(1)

    orig_height, orig_width = image.shape[:2]
    
    # Skalowanie w celu ujednolicenia detekcji
    det_width = 1500
    ratio = orig_width / det_width
    det_height = int(orig_height / ratio)
    resized = cv2.resize(image, (det_width, det_height))
    
    # 1. Przetwarzanie wstępne pod KĄTEM WYKRYWANIA KRAWĘDZI PAPIERU
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Użycie gradientu morfologicznego świetnie podkreśla krawędzie dokumentów
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # Binaryzacja i zamknięcie luk w krawędziach
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Canny dla uzyskania czystych, cienkich krawędzi
    edged = cv2.Canny(thresh, 50, 150)
    
    if args.debug:
        cv2.imwrite(os.path.join(args.output, "debug_edges.jpg"), edged)

    # 2. Znajdowanie konturów
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    det_area = det_width * det_height
    min_area = det_area * 0.05  # Paragon musi zajmować min. 5% obrazu
    accepted_receipts = [] 
    
    # 3. Filtrowanie i zatwierdzanie paragonów
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            break 
            
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), angle = rect
        
        if w == 0 or h == 0: continue
            
        rect_area = w * h
        # Solidność: Paragon powinien w miarę wypełniać swój obrócony prostokąt
        solidity = area / float(rect_area)
        if solidity < 0.65: continue
            
        # Proporcje (aspekt) paragonu: zazwyczaj od 1.3 do 6.0
        aspect = max(w, h) / min(w, h)
        if aspect < 1.3 or aspect > 6.0: continue
            
        # Pobierz 4 narożniki
        box = cv2.boxPoints(rect)
        x, y, w_bbox, h_bbox = cv2.boundingRect(c)
        curr_bbox = (x, y, w_bbox, h_bbox)
        
        # Sprawdź nakładanie się (IoU) z już znalezionymi paragonami
        is_duplicate = False
        for acc_box in accepted_receipts:
            iou = calculate_iou(curr_bbox, (acc_box[0], acc_box[1], acc_box[2], acc_box[3]))
            if iou > 0.4: 
                is_duplicate = True
                break
                
        if is_duplicate: continue
            
        # Przeskaluj do oryginalnej rozdzielczości
        orig_box = (box * ratio).astype(np.int32)
        orig_bbox = (int(x * ratio), int(y * ratio), int(w_bbox * ratio), int(h_bbox * ratio))
        accepted_receipts.append((*orig_bbox, orig_box))

    if not accepted_receipts:
        print("Błąd: Nie wykryto żadnych paragonów. Spróbuj użyć flagi --debug aby sprawdzić krawędzie.")
        sys.exit(1)

    # 4. Sortowanie w kolejności czytania
    sorted_receipts = sort_receipts_reading_order(accepted_receipts, orig_height)

    # 5. Ekstrakcja i zapis
    count = 0
    debug_img = image.copy() if args.debug else None
    
    for i, (x, y, w, h, box_pts) in enumerate(sorted_receipts):
        receipt_id = f"{i+1:03d}"
        
        try:
            unwarped = four_point_transform(image, box_pts)
            output_filename = os.path.join(args.output, f"receipt_{receipt_id}.jpg")
            cv2.imwrite(output_filename, unwarped)
            count += 1
            
            if args.debug:
                pts = box_pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(debug_img, [pts], True, (0, 255, 0), 4)
                cv2.putText(debug_img, f"#{receipt_id}", (x, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        except Exception as e:
            print(f"Błąd przetwarzania {receipt_id}: {e}")

    if args.debug:
        cv2.imwrite(os.path.join(args.output, "debug_boxes.jpg"), debug_img)
    
    print(f"Zakończono. Wyodrębniono {count} paragonów.")

if __name__ == "__main__":
    main()
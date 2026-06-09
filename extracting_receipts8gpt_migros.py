import cv2
import numpy as np
import argparse
import os
import sys

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def sort_receipts(receipts):
    return sorted(receipts, key=lambda r: (r[1] // 500, r[0]))

def detect_receipts(image, output_dir=None):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    _, mask = cv2.threshold(
        L,
        175,
        255,
        cv2.THRESH_BINARY
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )
    if output_dir:
        cv2.imwrite(
            os.path.join(output_dir, "debug_mask.jpg"),
            mask
        )
    mask_inv = cv2.bitwise_not(mask)
    if output_dir:
        cv2.imwrite(
            os.path.join(output_dir, "debug_mask_inv.jpg"),
            mask_inv
        )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_inv,
        connectivity=8
    )

    receipts = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50000:
            continue
        if w < 150:
            continue
        if h < 300:
            continue
        aspect = h / float(w)
        if aspect < 1.3:
            continue
        box = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)
        receipts.append((x, y, w, h, box))
        print(
            f"✓ receipt "
            f"x={x} y={y} "
            f"w={w} h={h} "
            f"area={area} "
            f"aspect={aspect:.2f}"
        )
    return receipts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    image = cv2.imread(args.input)
    if image is None:
        print("Nie można odczytać obrazu")
        sys.exit(1)
    receipts = detect_receipts(
        image,
        args.output if args.debug else None
    )
    if not receipts:
        print("Nie znaleziono paragonów")
        sys.exit(1)
    receipts = sort_receipts(receipts)
    debug_img = image.copy()
    for idx, (x, y, w, h, box) in enumerate(receipts):
        pts = box.astype(np.int32)
        cv2.rectangle(
            debug_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            4
        )
        cv2.putText(
            debug_img,
            f"#{idx+1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
    if args.debug:
        cv2.imwrite(
            os.path.join(args.output, "debug_final.jpg"),
            debug_img
        )
    count = 0
    for idx, (_, _, _, _, box) in enumerate(receipts):
        try:
            warped = four_point_transform(image, box)

            out_path = os.path.join(
                args.output,
                f"receipt_{idx+1:03d}.jpg"
            )
            cv2.imwrite(out_path, warped)
            print(f"Zapisano {out_path}")
            count += 1
        except Exception as e:
            print(e)
    print()
    print(f"Wykryto {count} paragonów")

if __name__ == "__main__":
    main()
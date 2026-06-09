import cv2
import numpy as np
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    image = cv2.imread(args.input)
    det_width = 1600
    ratio = image.shape[1] / det_width
    det_height = int(image.shape[0] / ratio)
    resized = cv2.resize(image, (det_width, det_height))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    blurred_t = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred_t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 51, 5)
    
    blurred_e = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred_e, 30, 100)
    combined = cv2.bitwise_or(thresh, edges)
    
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated_h = cv2.dilate(combined, h_kernel, iterations=1)
    
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    joined = cv2.dilate(dilated_h, v_kernel, iterations=1)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, kernel_close)
    
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = (det_width * det_height) * 0.002
    candidates = [c for c in cnts if cv2.contourArea(c) > min_area]
    
    print(f"Total contours: {len(cnts)}")
    print(f"Candidates (> {min_area:.1f} area): {len(candidates)}")
    
    for i, c in enumerate(candidates):
        x, y, w, h = cv2.boundingRect(c)
        print(f"Candidate {i}: x={x}, y={y}, w={w}, h={h}, area={cv2.contourArea(c)}")

if __name__ == "__main__":
    main()

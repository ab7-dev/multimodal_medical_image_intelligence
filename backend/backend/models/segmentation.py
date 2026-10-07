"""
Advanced Region of Interest (ROI) Segmentation & Anatomical Localization Engine.
Extracts lesion contours, bounding boxes, and maps findings to anatomical lung zones.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from backend.core.config import ANATOMICAL_ZONES

class LesionSegmenter:
    """Segments problem areas from activation heatmaps and computes geometric & anatomical properties."""

    def __init__(self, activation_threshold: float = 0.45):
        self.activation_threshold = activation_threshold

    def segment_pathology_roi(
        self,
        heatmap_2d: np.ndarray,
        orig_image_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Extracts segmented contours, bounding boxes, area, and anatomical zone from the heatmap.
        orig_image_shape: (height, width)
        """
        orig_h, orig_w = orig_image_shape[:2]
        
        # If heatmap has zero or very low activation, no visual grounding exists
        if heatmap_2d is None or np.max(heatmap_2d) < 0.25:
            return {
                "has_visual_grounding": False,
                "primary_region": None,
                "all_regions": [],
                "region_count": 0,
                "binary_mask_summary": {"active_pixels": 0, "coverage_pct": 0.0}
            }

        # Resize heatmap to original image dimensions
        resized_cam = cv2.resize(heatmap_2d, (orig_w, orig_h))
        
        # Dynamic thresholding: Otsu or minimum cutoff
        cam_uint8 = np.uint8(255 * resized_cam)
        
        # Adaptive cutoff: max(threshold, mean + 0.5 * std)
        mean_val = np.mean(resized_cam)
        std_val = np.std(resized_cam)
        cutoff = max(self.activation_threshold, float(mean_val + 0.7 * std_val))
        
        _, binary_mask = cv2.threshold(
            cam_uint8,
            int(cutoff * 255),
            255,
            cv2.THRESH_BINARY
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter small noise artifacts
        min_contour_area = 0.005 * (orig_h * orig_w)  # at least 0.5% of image
        valid_regions = []

        total_image_area = float(orig_h * orig_w)

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_contour_area:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Contour polygon points normalized [0, 1]
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            polygon_pts = [[round(float(pt[0][0]) / orig_w, 4), round(float(pt[0][1]) / orig_h, 4)] for pt in approx]

            # Centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"]) / orig_w
                cy = float(M["m01"] / M["m00"]) / orig_h
            else:
                cx = (x + w / 2) / orig_w
                cy = (y + h / 2) / orig_h

            # Identify matching anatomical zone
            matched_zone = self._match_anatomical_zone(cx, cy, [y/orig_h, x/orig_w, (y+h)/orig_h, (x+w)/orig_w])

            area_pct = round((area / total_image_area) * 100.0, 2)

            valid_regions.append({
                "bounding_box": {
                    "ymin": round(y / orig_h, 4),
                    "xmin": round(x / orig_w, 4),
                    "ymax": round((y + h) / orig_h, 4),
                    "xmax": round((x + w) / orig_w, 4),
                    "pixel_x": int(x),
                    "pixel_y": int(y),
                    "pixel_w": int(w),
                    "pixel_h": int(h)
                },
                "centroid": {"x": round(cx, 4), "y": round(cy, 4)},
                "area_percent": area_pct,
                "polygon": polygon_pts,
                "anatomical_zone": matched_zone
            })

        # Sort regions by area descending
        valid_regions.sort(key=lambda r: r["area_percent"], reverse=True)

        primary_roi = valid_regions[0] if valid_regions else None
        
        has_visual_grounding = primary_roi is not None

        return {
            "has_visual_grounding": has_visual_grounding,
            "primary_region": primary_roi,
            "all_regions": valid_regions,
            "region_count": len(valid_regions),
            "binary_mask_summary": {
                "active_pixels": int(np.count_nonzero(binary_mask)),
                "coverage_pct": round((np.count_nonzero(binary_mask) / total_image_area) * 100.0, 2)
            }
        }

    def _match_anatomical_zone(self, cx: float, cy: float, box: List[float]) -> str:
        """Determines anatomical region based on lesion centroid coordinates."""
        ymin, xmin, ymax, xmax = box
        
        best_zone = "Unspecified Hemithorax"
        best_score = -1.0

        for zone in ANATOMICAL_ZONES:
            zy1, zx1, zy2, zx2 = zone["box"]
            # Check if centroid is inside
            inside = (zy1 <= cy <= zy2) and (zx1 <= cx <= zx2)
            if inside:
                return zone["name"]

            # Compute IoU/overlap score
            inter_y1 = max(ymin, zy1)
            inter_x1 = max(xmin, zx1)
            inter_y2 = min(ymax, zy2)
            inter_x2 = min(xmax, zx2)

            inter_area = max(0.0, inter_y2 - inter_y1) * max(0.0, inter_x2 - inter_x1)
            if inter_area > best_score:
                best_score = inter_area
                best_zone = zone["name"]

        return best_zone

    @staticmethod
    def draw_annotations(
        orig_image_bgr: np.ndarray,
        segmentation_result: Dict[str, Any],
        pathology_name: str,
        confidence_pct: float
    ) -> np.ndarray:
        """Draws highlighted lesion contours and bounding boxes onto the image."""
        annotated = orig_image_bgr.copy()
        h, w = annotated.shape[:2]

        regions = segmentation_result.get("all_regions", [])
        for i, reg in enumerate(regions):
            bbox = reg["bounding_box"]
            px = bbox["pixel_x"]
            py = bbox["pixel_y"]
            pw = bbox["pixel_w"]
            ph = bbox["pixel_h"]

            # Draw glowing bounding box
            color = (0, 165, 255) if i == 0 else (255, 100, 0)  # Orange for primary
            cv2.rectangle(annotated, (px, py), (px + pw, py + ph), color, 2)

            # Draw polygon contour if available
            polygon_pts = reg.get("polygon", [])
            if len(polygon_pts) >= 3:
                pts_arr = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon_pts], np.int32)
                pts_arr = pts_arr.reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts_arr], True, (0, 255, 200), 2)

            # Label banner
            zone = reg.get("anatomical_zone", "ROI")
            label_text = f"{pathology_name} ({confidence_pct:.1f}%) | {zone}"
            
            # Text background
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (px, max(0, py - th - 8)), (px + tw + 6, py), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (px + 3, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return annotated

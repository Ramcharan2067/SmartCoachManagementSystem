"""
Edge Node AI Processor (Instance Segmentation Engine)
-----------------------------------------------------
Utilizes YOLOv8 Segmentation to draw pixel-perfect contours around humans.
Calculates density by measuring the exact percentage of human pixels 
in the walkable region of interest, entirely eliminating bounding box "empty air" inflation.
"""

from ultralytics import YOLO
import cv2
import requests
import os
import time
import numpy as np

SERVER_URL = "http://127.0.0.1:5000"
SAVE_DIR = "static/captures"
COACHES = ["GEN-1", "GEN-2", "GEN-3", "GEN-4", "SLR-1", "SLR-2", "SLR-M"]

def calculate_segmentation_density(img, results):
    """
    Calculates density using exact pixel masks.
    """
    h, w = img.shape[:2]
    
    # If no people detected, or no masks generated
    if len(results[0].boxes) == 0 or results[0].masks is None:
        return 0, img.copy()

    # 1. Extract Segmentation Masks
    # Masks are output as a tensor of shape (N, H, W). We combine them into one flat 2D mask.
    masks = results[0].masks.data.cpu().numpy()
    
    # Combine all individual human masks into a single binary mask (255 = Human, 0 = Background)
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    
    # YOLO output masks are slightly lower resolution, resize back to original image size
    combined_mask = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. Define Region of Interest (ROI) - The Walkable Space
    # Ignore the top 30% of the image (ceiling, fans)
    ceiling_cutoff = int(h * 0.30)
    roi_mask = combined_mask[ceiling_cutoff:h, 0:w]
    
    occupied_human_pixels = cv2.countNonZero(roi_mask)
    total_roi_pixels = w * (h - ceiling_cutoff)
    
    pixel_fill_ratio = occupied_human_pixels / total_roi_pixels

    # 3. Calibration
    # A perfectly crush-loaded train physically covers about 50% of the 2D image pixels in the ROI
    # due to perspective (floor and walls will always partially show).
    MAX_FILL_RATIO = 0.50 
    
    density = int((pixel_fill_ratio / MAX_FILL_RATIO) * 100)
    density = min(max(density, 0), 100) 

    # 4. Visual Proof Generation
    visual_img = img.copy()
    
    # Create a translucent color overlay for the human masks (Green for safe, Red for crowded)
    color_overlay = np.zeros_like(visual_img)
    mask_color = [0, 0, 255] if density >= 85 else [0, 255, 0] # BGR Format
    color_overlay[:, :] = mask_color
    
    # Apply the combined mask to the original image
    colored_mask = cv2.bitwise_and(color_overlay, color_overlay, mask=combined_mask)
    cv2.addWeighted(colored_mask, 0.6, visual_img, 1.0, 0, visual_img)
    
    # Draw ROI boundary line
    cv2.line(visual_img, (0, ceiling_cutoff), (w, ceiling_cutoff), (0, 255, 255), 2)
    
    # Add Telemetry text
    status_color = (0, 0, 255) if density >= 85 else ((0, 255, 255) if density >= 50 else (0, 255, 0))
    cv2.putText(visual_img, f"Human Pixel Area: {int(pixel_fill_ratio*100)}%", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(visual_img, f"Calculated Density: {density}%", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    
    return density, visual_img

def edge_process():
    print("[SYSTEM] Initializing Edge AI Node (Segmentation Engine)...")
    # Switched to Segmentation Model
    model = YOLO("yolov8s-seg.pt") 
    
    if not os.path.exists(SAVE_DIR): 
        os.makedirs(SAVE_DIR)

    last_processed_state = None

    while True:
        try:
            res = requests.get(f"{SERVER_URL}/api/get_live_status")
            state = res.json()
            
            if not state.get('active_train'):
                time.sleep(2)
                continue

            current_state_key = f"{state['active_train']}_{state['station_index']}"
            if current_state_key == last_processed_state:
                time.sleep(2)
                continue

            station_idx = state['station_index']
            print(f"[EVENT] Train departed. Initiating Segmentation Analysis...")
            
            folder_path = f"test_images/stop_{station_idx}"
            if not os.path.exists(folder_path):
                folder_path = "test_images"

            for coach in COACHES:
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    temp_path = os.path.join(folder_path, f"{coach}{ext}")
                    if os.path.exists(temp_path):
                        img_path = temp_path
                        break
                
                if not img_path: 
                    continue

                img = cv2.imread(img_path)
                
                # Run prediction strictly looking for class 0 (Person)
                results = model.predict(img, classes=[0], conf=0.25, verbose=False)
                
                density, visual_img = calculate_segmentation_density(img, results)
                
                save_path = os.path.join(SAVE_DIR, f"{coach}_live.jpg")
                cv2.imwrite(save_path, visual_img)
                
                requests.post(f"{SERVER_URL}/api/update_crowd", json={"coach_id": coach, "crowd_percent": density})
                
            last_processed_state = current_state_key
            print("[SYSTEM] Edge analysis complete. Entering standby mode.")
                
        except Exception as e: 
            pass 
            
        time.sleep(2)

if __name__ == "__main__":
    edge_process()
import os
import cv2
import numpy as np
from ultralytics import YOLOE  # (또는 YOLO)

model = YOLOE("yoloe-26x-seg.pt")
model.set_classes(["person"])

source_folder = "./metro_pure_haze_inputs/metro_pure_predictions"
attention_output_folder = "./crowd_color_grid"
os.makedirs(attention_output_folder, exist_ok=True)

GRID_ROWS = 15
GRID_COLS = 20

image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


for img_file in image_files:
    img_path = os.path.join(source_folder, img_file)
    results = model.predict(source=img_path, save=False, verbose=False)
    result = results[0]

    orig_img = result.orig_img.copy()
    img_h, img_w = orig_img.shape[:2]

    density_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS

    if len(result.boxes) > 0:
        centers = result.boxes.xywh[:, :2].cpu().numpy()
        for cx, cy in centers:
            col_idx = min(int(cx / cell_w), GRID_COLS - 1)
            row_idx = min(int(cy / cell_h), GRID_ROWS - 1)
            density_map[row_idx, col_idx] += 1

    max_density = np.max(density_map)
    if max_density > 0:
        norm_map = (density_map / max_density) * 255
    else:
        norm_map = density_map

    norm_map = norm_map.astype(np.uint8)

    heatmap_resized = cv2.resize(norm_map, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    mask = (heatmap_resized > 0)[:, :, np.newaxis]
    alpha = 0.5  # 색상 투명도

    final_img = np.where(mask, cv2.addWeighted(orig_img, 1 - alpha, heatmap_color, alpha, 0), orig_img)

    for r in range(1, GRID_ROWS):
        y = int(r * cell_h)
        cv2.line(final_img, (0, y), (img_w, y), (255, 255, 255), 1, cv2.LINE_AA)
    for c in range(1, GRID_COLS):
        x = int(c * cell_w)
        cv2.line(final_img, (x, 0), (x, img_h), (255, 255, 255), 1, cv2.LINE_AA)

    save_path = os.path.join(attention_output_folder, img_file)
    cv2.imwrite(save_path, final_img)

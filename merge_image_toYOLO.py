import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

# 合併用工具 ──────────────────────────────────────
def is_near_patch_edge(box, img_width, img_height, crop_width, crop_height, edge_threshold=20):
    x_center, y_center, width, height = box[1], box[2], box[3], box[4]
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    patch_col = int(x_center // crop_width)
    patch_row = int(y_center // crop_height)
    patch_x_min = patch_col * crop_width
    patch_y_min = patch_row * crop_height
    patch_x_max = patch_x_min + crop_width
    patch_y_max = patch_y_min + crop_height

    near_left   = (x_min - patch_x_min) < edge_threshold
    near_right  = (patch_x_max - x_max) < edge_threshold
    near_top    = (y_min - patch_y_min) < edge_threshold
    near_bottom = (patch_y_max - y_max) < edge_threshold

    return near_left or near_right or near_top or near_bottom

def boxes_are_adjacent(box1, box2, max_distance=20):
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    horizontal_adjacent = (abs(x1_min - x2_max) < max_distance or abs(x1_max - x2_min) < max_distance)
    vertical_overlap = not (y1_max < y2_min or y2_max < y1_min)

    vertical_adjacent = (abs(y1_min - y2_max) < max_distance or abs(y1_max - y2_min) < max_distance)
    horizontal_overlap = not (x1_max < x2_min or x2_max < x1_min)

    return (horizontal_adjacent and vertical_overlap) or (vertical_adjacent and horizontal_overlap)

def merge_two_boxes(box1, box2):
    cls = box1[0]
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    x_min = min(x1_min, x2_min)
    y_min = min(y1_min, y2_min)
    x_max = max(x1_max, x2_max)
    y_max = max(y1_max, y2_max)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    confidence = max(box1[5], box2[5])

    return [cls, x_center, y_center, width, height, confidence]

def merge_boxes_across_patches(boxes, img_width, img_height, crop_width, crop_height, edge_threshold=20):
    merged_boxes = []
    used_indices = set()
    position_to_indices = {}

    for idx, box in enumerate(boxes):
        x_center, y_center = box[1], box[2]
        patch_col = int(x_center // crop_width)
        patch_row = int(y_center // crop_height)
        key = (patch_row, patch_col)
        position_to_indices.setdefault(key, []).append(idx)

    for idx, box in enumerate(boxes):
        if idx in used_indices:
            continue

        cls = box[0]
        merged_box = box.copy()
        x_center, y_center = box[1], box[2]
        patch_col = int(x_center // crop_width)
        patch_row = int(y_center // crop_height)

        if is_near_patch_edge(box, img_width, img_height, crop_width, crop_height, edge_threshold):
            neighbors = [
                (patch_row + dr, patch_col + dc)
                for dr in [-1, 0, 1] for dc in [-1, 0, 1] if not (dr == 0 and dc == 0)
            ]
            for neighbor in neighbors:
                for n_idx in position_to_indices.get(neighbor, []):
                    if n_idx in used_indices:
                        continue
                    neighbor_box = boxes[n_idx]
                    if neighbor_box[0] != cls:
                        continue
                    if boxes_are_adjacent(merged_box, neighbor_box, max_distance=edge_threshold):
                        merged_box = merge_two_boxes(merged_box, neighbor_box)
                        used_indices.add(n_idx)

        merged_boxes.append(merged_box)
        used_indices.add(idx)

    return merged_boxes

# 拼接 patch 並進行標註整合 ──────────────────────────────────────────
def collect_all_patches(image_root):
    patch_map = {}
    for root, _, files in os.walk(image_root):
        for fname in files:
            if '_cropped_' not in fname or not fname.lower().endswith(('.jpg', '.png')):
                continue
            basename, ext = os.path.splitext(fname)
            parts = basename.split('_cropped_')
            if len(parts) != 2:
                continue
            original_name = parts[0]
            try:
                row, col = map(int, parts[1].split('_'))
            except ValueError:
                continue
            patch_map.setdefault(original_name, []).append((row, col, os.path.join(root, fname)))
    return patch_map

def merge_patches_all(image_root, label_root, output_dir,
                      img_width, img_height, crop_width, crop_height, edge_threshold=20):
    patch_all_images_dir = os.path.join(output_dir, "images")
    patch_all_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(patch_all_images_dir, exist_ok=True)
    os.makedirs(patch_all_labels_dir, exist_ok=True)

    patch_map = collect_all_patches(image_root)
    expected_rows = img_height // crop_height
    expected_cols = img_width // crop_width

    for original_name in tqdm(sorted(patch_map.keys()), desc="🧩 拼接與合併", ncols=100):
        canvas = 255 * np.ones((img_height, img_width, 3), dtype=np.uint8)
        all_boxes = []
        patches = patch_map[original_name]
        patch_positions = {(r, c): path for r, c, path in patches}

        for r in range(expected_rows):
            for c in range(expected_cols):
                if (r, c) not in patch_positions:
                    continue
                patch_img_path = patch_positions[(r, c)]
                x_offset, y_offset = c * crop_width, r * crop_height

                img = cv2.imread(patch_img_path)
                if img is not None:
                    canvas[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width] = img

                label_path = patch_img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            cls, xc, yc, w, h = map(float, parts[:5])
                            x1 = (xc - w / 2) * crop_width + x_offset
                            y1 = (yc - h / 2) * crop_height + y_offset
                            x2 = (xc + w / 2) * crop_width + x_offset
                            y2 = (yc + h / 2) * crop_height + y_offset
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            all_boxes.append([cls, x_center, y_center, width, height, 1.0])

        # 合併跨 patch 框
        merged_boxes = merge_boxes_across_patches(all_boxes, img_width, img_height, crop_width, crop_height, edge_threshold)

        # 儲存拼接圖片
        cv2.imwrite(os.path.join(patch_all_images_dir, original_name + ".jpg"), canvas)

        # 儲存標註
        output_label_path = os.path.join(patch_all_labels_dir, original_name + ".txt")
        with open(output_label_path, 'w') as f:
            for box in merged_boxes:
                cls, xc, yc, w, h, _ = box
                f.write(f"{int(cls)} {xc / img_width:.6f} {yc / img_height:.6f} {w / img_width:.6f} {h / img_height:.6f}\n")

# === 執行設定 ===
image_root = "D:\\Github\\RandomPick_v6"
label_root = "D:\\Github\\RandomPick_v6"
output_dir = "D:\\Github\\RandomPick_v6_Combined"

merge_patches_all(
    image_root=image_root,
    label_root=label_root,
    output_dir=output_dir,
    img_width=2560,
    img_height=1920,
    crop_width=640,
    crop_height=640,
    edge_threshold=5  # 合併敏感度可調整
)

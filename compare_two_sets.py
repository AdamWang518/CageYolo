# compare_yolo_models_patchmerge.py
"""
Compare YOLO patch‑based and full‑image models, but with an improved
patch pipeline that merges detections which straddle neighbouring patches.

Key changes vs. original `compare_yolo_models.py`
-------------------------------------------------
* Added edge‑detection & adjacency helpers (`is_near_patch_edge`,
  `boxes_are_adjacent`, `merge_two_boxes`).
* Added `merge_boxes_across_patches` which merges NMS‑filtered detections
  that belong to the same physical object but were split across two
  adjacent patches.
* `process_patch` pipeline now performs:
      ├─ raw inference per patch → collect boxes
      ├─ class‑wise NMS            (remove dup. detections)
      └─ cross‑patch merge         (produce final boxes)
* Evaluation and full‑model path remain unchanged.

The script is self‑contained – just adjust the constants section to match
local paths/thresholds and run `python compare_yolo_models_patchmerge.py`.
"""

import os
import cv2
import torch
import csv
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.ops import nms

# ───────────────────  基本設定  ───────────────────
PATCH_MODEL_PATH = r"runs_patch\\exp_patch\\weights\\best.pt"          # patch‑cut模型
FULL_MODEL_PATH  = r"runs_full\\exp_full\\weights\\best.pt"           # 整圖模型
INPUT_IMG_DIR    = r"D:\\Github\\RandomPick_v6_6_Full\\val\\images"
GT_LABEL_DIR     = r"D:\\Github\\RandomPick_v6_6_Full\\val\\labels"
OUTPUT_PATCH_DIR = r"D:\\Github\\CompareResult\\output_patch_F"
OUTPUT_FULL_DIR  = r"D:\\Github\\CompareResult\\output_full_F"
CLASS_NAMES      = ['ship', 'aquaculture cage', 'buoy']
CONF_THRES       = 0.5
IOU_THRES        = 0.5
EDGE_THRES       = 5          # 只允許「左右跨一個 patch」的框被合併
CROP_W = CROP_H  = 640        # patch 大小
# ──────────────────────────────────────────────── #
# ─────────────────── 通用工具 ───────────────────

def xywhn_to_xyxy(box):
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

def calc_iou(box1, box2):
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)

def draw_boxes_on_image(img, boxes, colors, names):
    for cls, x, y, w, h, conf in boxes:
        x1, y1 = int((x - w / 2) * img.shape[1]), int((y - h / 2) * img.shape[0])
        x2, y2 = int((x + w / 2) * img.shape[1]), int((y + h / 2) * img.shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(cls, (255, 255, 255)), 2)
        cv2.putText(img, f"{names[cls]} {conf:.2f}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.get(cls, (255, 255, 255)), 1)

# ─────────────────── Patch Merge Helpers ───────────────────

def is_near_patch_edge(box, img_w, img_h, crop_w, crop_h, thr=EDGE_THRES):
    x_c, y_c, w, h = box[1:5]
    x_min, y_min = x_c - w / 2, y_c - h / 2
    x_max, y_max = x_c + w / 2, y_c + h / 2

    col = int(x_c // crop_w)
    row = int(y_c // crop_h)
    patch_x_min, patch_y_min = col * crop_w, row * crop_h
    patch_x_max, patch_y_max = patch_x_min + crop_w, patch_y_min + crop_h

    return (
        (x_min - patch_x_min) < thr or (patch_x_max - x_max) < thr or
        (y_min - patch_y_min) < thr or (patch_y_max - y_max) < thr
    )

def boxes_are_adjacent(b1, b2, max_dist=EDGE_THRES):
    x1_min, y1_min = b1[1] - b1[3] / 2, b1[2] - b1[4] / 2
    x1_max, y1_max = b1[1] + b1[3] / 2, b1[2] + b1[4] / 2
    x2_min, y2_min = b2[1] - b2[3] / 2, b2[2] - b2[4] / 2
    x2_max, y2_max = b2[1] + b2[3] / 2, b2[2] + b2[4] / 2

    horiz_touch = abs(x1_min - x2_max) < max_dist or abs(x1_max - x2_min) < max_dist
    vert_overlap = not (y1_max < y2_min or y2_max < y1_min)
    vert_touch = abs(y1_min - y2_max) < max_dist or abs(y1_max - y2_min) < max_dist
    horiz_overlap = not (x1_max < x2_min or x2_max < x1_min)

    return (horiz_touch and vert_overlap) or (vert_touch and horiz_overlap)

def merge_two_boxes(b1, b2):
    cls = b1[0]
    x_min = min(b1[1] - b1[3] / 2, b2[1] - b2[3] / 2)
    y_min = min(b1[2] - b1[4] / 2, b2[2] - b2[4] / 2)
    x_max = max(b1[1] + b1[3] / 2, b2[1] + b2[3] / 2)
    y_max = max(b1[2] + b1[4] / 2, b2[2] + b2[4] / 2)
    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    conf = min(b1[5], b2[5])
    return [cls, x_c, y_c, w, h, conf]

def merge_boxes_across_patches(boxes, img_w, img_h, crop_w, crop_h, thr=EDGE_THRES):
    used = set()
    merged = []
    patch_map = {}
    for idx, b in enumerate(boxes):
        col = int(b[1] // crop_w)
        row = int(b[2] // crop_h)
        patch_map.setdefault((row, col), []).append(idx)

    for idx, b in enumerate(boxes):
        if idx in used:
            continue
        cur = b.copy()
        if is_near_patch_edge(b, img_w, img_h, crop_w, crop_h, thr):
            col = int(b[1] // crop_w)
            row = int(b[2] // crop_h)
            neighbours = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
            for nb in neighbours:
                for n_idx in patch_map.get(nb, []):
                    if n_idx in used:
                        continue
                    nb_box = boxes[n_idx]
                    if nb_box[0] != b[0]:
                        continue
                    if boxes_are_adjacent(cur, nb_box, max_dist=thr):
                        cur = merge_two_boxes(cur, nb_box)
                        used.add(n_idx)
        merged.append(cur)
        used.add(idx)
    return merged


# ---------- Model‑A：Patch 推論 ----------

def process_patch(model, img_dir, out_dir, names):
    os.makedirs(f"{out_dir}/labels", exist_ok=True)
    os.makedirs(f"{out_dir}/images", exist_ok=True)

    for p in tqdm([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))], desc="Patch model"):
        img = cv2.imread(os.path.join(img_dir, p))
        if img is None:
            continue
        h, w = img.shape[:2]
        raw_boxes = []

        # patch inference
        for r in range(h // CROP_H):
            for c in range(w // CROP_W):
                x0, y0 = c * CROP_W, r * CROP_H
                patch = img[y0:y0 + CROP_H, x0:x0 + CROP_W]
                for res in model(patch, verbose=False):
                    for b in res.boxes:
                        cls = int(b.cls)
                        x, y, bw, bh = b.xywh[0].tolist()
                        conf = b.conf.item()
                        if conf < CONF_THRES:
                            continue
                        # map back to full‑img absolute px
                        x_full = x0 + x
                        y_full = y0 + y
                        raw_boxes.append([cls, x_full, y_full, bw, bh, conf])

        if not raw_boxes:
            continue

        # 1) class‑wise NMS on absolute coords
        nms_boxes = []
        for cls in set(b[0] for b in raw_boxes):
            cls_boxes = [b for b in raw_boxes if b[0] == cls]
            boxes_xyxy = torch.tensor([[b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2] for b in cls_boxes])
            confs = torch.tensor([b[5] for b in cls_boxes])
            keep = nms(boxes_xyxy, confs, IOU_THRES)
            nms_boxes.extend([cls_boxes[i] for i in keep])

        # 2) merge across adjacent patches
        merged = merge_boxes_across_patches(nms_boxes, w, h, CROP_W, CROP_H, EDGE_THRES)

        # convert back to *normalized* xywh for saving
        final = []
        for cls, x_px, y_px, bw_px, bh_px, conf in merged:
            final.append([
                cls,
                x_px / w,
                y_px / h,
                bw_px / w,
                bh_px / h,
                conf,
            ])

        name = os.path.splitext(p)[0]
        with open(f"{out_dir}/labels/{name}.txt", "w") as f:
            for cls, x, y, bw, bh, conf in final:
                f.write(f"{int(cls)} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")


        img_draw = img.copy()
        draw_boxes_on_image(img_draw, final, {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}, names)
        cv2.imwrite(f"{out_dir}/images/{name}.jpg", img_draw)


# ---------- Model‑B：整圖推論（原始程式） ----------

def process_full(model, img_dir, out_dir, names):
    os.makedirs(f"{out_dir}/labels", exist_ok=True)
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    for p in tqdm([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))], desc="Full model"):
        img = cv2.imread(os.path.join(img_dir, p))
        if img is None:
            continue
        h, w = img.shape[:2]
        final = []
        for res in model(img, verbose=False)[0].boxes:
            cls = int(res.cls)
            x, y, bw, bh = res.xywhn[0].tolist()
            conf = res.conf.item()
            if conf >= CONF_THRES:
                final.append([cls, x, y, bw, bh, conf])
        name = os.path.splitext(p)[0]
        with open(f"{out_dir}/labels/{name}.txt", "w") as f:
            for cls, x, y, bw, bh, conf in final:
                f.write(f"{int(cls)} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")
        img_draw = img.copy()
        draw_boxes_on_image(img_draw, final, {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}, names)
        cv2.imwrite(f"{out_dir}/images/{name}.jpg", img_draw)


# ---------- 統計比較（原始 evaluate() 不變） ----------

def evaluate(pred_dir, gt_dir, class_num, iou_thr=0.5):
    TP, FP, FN = [0] * class_num, [0] * class_num, [0] * class_num
    for lbl in os.listdir(gt_dir):
        if not lbl.endswith(".txt"):
            continue
        gt_boxes = {i: [] for i in range(class_num)}
        with open(os.path.join(gt_dir, lbl)) as f:
            for line in f:
                try:
                    c, x, y, w, h, *_ = [float(t) for t in line.split()]
                except ValueError:
                    continue
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    continue
                gt_boxes[int(c)].append(xywhn_to_xyxy([x, y, w, h]))

        pred_path = os.path.join(pred_dir, lbl)
        pred_boxes = {i: [] for i in range(class_num)}
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                for line in f:
                    try:
                        c, x, y, w, h, *_ = [float(t) for t in line.split()]
                    except ValueError:
                        continue
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        continue
                    pred_boxes[int(c)].append(xywhn_to_xyxy([x, y, w, h]))

        for cls in range(class_num):
            gts, preds = gt_boxes[cls], pred_boxes[cls]
            matched = [False] * len(preds)
            for g in gts:
                hit = False
                for idx, p in enumerate(preds):
                    if not matched[idx] and calc_iou(g, p) >= iou_thr:
                        TP[cls] += 1
                        matched[idx] = True
                        hit = True
                        break
                if not hit:
                    FN[cls] += 1
            FP[cls] += matched.count(False)
    return TP, FP, FN

# ────────────────── 主程式 ──────────────────
if __name__ == "__main__":
    patch_model = YOLO(PATCH_MODEL_PATH)
    full_model = YOLO(FULL_MODEL_PATH)

    process_patch(patch_model, INPUT_IMG_DIR, OUTPUT_PATCH_DIR, CLASS_NAMES)
    process_full(full_model, INPUT_IMG_DIR, OUTPUT_FULL_DIR, CLASS_NAMES)

    TP_p, FP_p, FN_p = evaluate(f"{OUTPUT_PATCH_DIR}/labels", GT_LABEL_DIR, len(CLASS_NAMES))
    TP_f, FP_f, FN_f = evaluate(f"{OUTPUT_FULL_DIR}/labels", GT_LABEL_DIR, len(CLASS_NAMES))

    # ----------- 輸出統計結果 -----------
    header = ["Class", "GT", "Patch TP", "Patch Recall", "Full TP", "Full Recall"]
    rows = []
    print("\n=====  Detection Results vs. Ground‑Truth  =====")
    for i, name in enumerate(CLASS_NAMES):
        gt = TP_p[i] + FN_p[i]
        rec_p = TP_p[i] / gt if gt else 0
        rec_f = TP_f[i] / gt if gt else 0
        rows.append([name, gt, TP_p[i], f"{rec_p:.3f}", TP_f[i], f"{rec_f:.3f}"])
        print(f"{name:20s} | GT:{gt:4d} | Patch TP:{TP_p[i]:4d} R:{rec_p:.3f} | Full TP:{TP_f[i]:4d} R:{rec_f:.3f}")

    with open("compare_stats.csv", "w", newline="") as csvfile:
        csv.writer(csvfile).writerows([header] + rows)
    print("\n📊 統計已儲存 compare_stats.csv")

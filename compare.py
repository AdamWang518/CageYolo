# compare_yolo_models.py
import os, cv2, torch
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.ops import nms
import csv

# ───────────────────  基本設定  ───────────────────
PATCH_MODEL_PATH = "patch\\weights\\best.pt"
FULL_MODEL_PATH  = "full\\weights\\best.pt"        # ← 改成 best.pt
INPUT_IMG_DIR    = r"D:\\Github\\RandomPick_v6_5_Combined\\test\\images"
GT_LABEL_DIR     = r"D:\\Github\\RandomPick_v6_5_Combined\\test\\labels"
OUTPUT_PATCH_DIR = "D:\Github\CompareResult\\output_patch"                  # 產出目錄
OUTPUT_FULL_DIR  = "D:\Github\CompareResult\\output_full"
CLASS_NAMES      = ['ship', 'aquaculture cage', 'buoy']
CONF_THRES       = 0.5
IOU_THRES        = 0.5
CROP_W = CROP_H  = 640   # patch 大小
# ──────────────────────────────────────────────── #

# ---------- 通用小工具 ----------
def xywhn_to_xyxy(box):
    """YOLO normalized xywh → xyxy (同樣以 0‑1 座標計)"""
    x, y, w, h = box
    return [x-w/2, y-h/2, x+w/2, y+h/2]

def calc_iou(box1, box2):
    """IoU on normalized xyxy"""
    xa, ya, xb, yb = max(box1[0], box2[0]), max(box1[1], box2[1]), \
                     min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    if inter == 0: return 0.0
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

# ---------- 畫框（用於視覺化，可略） ----------
def draw_boxes_on_image(img, boxes, colors, names):
    for cls, x, y, w, h, conf in boxes:
        x1, y1 = int((x-w/2)*img.shape[1]), int((y-h/2)*img.shape[0])
        x2, y2 = int((x+w/2)*img.shape[1]), int((y+h/2)*img.shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(cls,(255,255,255)), 2)
        cv2.putText(img, f"{names[cls]} {conf:.2f}", (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.get(cls,(255,255,255)), 1)

# ---------- Model‑A：Patch 推論 ----------
def process_patch(model, img_dir, out_dir, names):
    os.makedirs(f"{out_dir}/labels", exist_ok=True)
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    for p in tqdm([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))], desc="Patch model"):
        img = cv2.imread(os.path.join(img_dir, p)); h, w = img.shape[:2]
        all_boxes = []
        for r in range(h // CROP_H):
            for c in range(w // CROP_W):
                x0, y0 = c * CROP_W, r * CROP_H
                patch = img[y0:y0 + CROP_H, x0:x0 + CROP_W]
                for res in model(patch, verbose=False):
                    for b in res.boxes:
                        cls = int(b.cls); x, y, bw, bh = b.xywhn[0].tolist()
                        conf = b.conf.item()
                        if conf >= CONF_THRES:
                            x = (x0 + x * CROP_W) / w
                            y = (y0 + y * CROP_H) / h
                            bw *= CROP_W / w
                            bh *= CROP_H / h
                            all_boxes.append([cls, x, y, bw, bh, conf])
        final = []
        for cls in set(b[0] for b in all_boxes):
            cls_boxes = [b for b in all_boxes if b[0] == cls]
            if not cls_boxes: continue
            xyxy = torch.tensor([xywhn_to_xyxy(b[1:5]) for b in cls_boxes])
            confs = torch.tensor([b[5] for b in cls_boxes])
            keep = nms(xyxy, confs, IOU_THRES)
            final += [cls_boxes[i] for i in keep]
        name = os.path.splitext(p)[0]
        with open(f"{out_dir}/labels/{name}.txt", 'w') as f:
            for cls, x, y, bw, bh, conf in final:
                f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")
        # ⬇️ 畫圖輸出
        img_draw = img.copy()
        draw_boxes_on_image(img_draw, final, {0:(0,255,0),1:(0,0,255),2:(255,0,0)}, names)
        cv2.imwrite(f"{out_dir}/images/{name}.jpg", img_draw)

# ---------- Model‑B：整圖推論 ----------
def process_full(model, img_dir, out_dir, names):
    os.makedirs(f"{out_dir}/labels", exist_ok=True)
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    for p in tqdm([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))], desc="Full model"):
        img = cv2.imread(os.path.join(img_dir, p)); h, w = img.shape[:2]
        final = []
        for res in model(img, verbose=False)[0].boxes:
            cls = int(res.cls); x, y, bw, bh = res.xywhn[0].tolist(); conf = res.conf.item()
            if conf >= CONF_THRES:
                final.append([cls, x, y, bw, bh, conf])
        name = os.path.splitext(p)[0]
        with open(f"{out_dir}/labels/{name}.txt", 'w') as f:
            for cls, x, y, bw, bh, conf in final:
                f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")
        # ⬇️ 畫圖輸出
        img_draw = img.copy()
        draw_boxes_on_image(img_draw, final, {0:(0,255,0),1:(0,0,255),2:(255,0,0)}, names)
        cv2.imwrite(f"{out_dir}/images/{name}.jpg", img_draw)

# ---------- 統計比較 ----------
def evaluate(pred_dir, gt_dir, class_num, iou_thr=0.5):
    TP=[0]*class_num; FP=[0]*class_num; FN=[0]*class_num
    for lbl in os.listdir(gt_dir):
        if not lbl.endswith('.txt'): continue
        # 讀 GT
        gt_boxes={i:[] for i in range(class_num)}
        with open(os.path.join(gt_dir,lbl)) as f:
            for line in f:
                c,x,y,w,h,*_=[float(t) for t in line.split()]
                gt_boxes[int(c)].append(xywhn_to_xyxy([x,y,w,h]))
        # 讀 Pred
        pred_path=os.path.join(pred_dir,lbl)
        pred_boxes={i:[] for i in range(class_num)}
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                for line in f:
                    c,x,y,w,h,*_=[float(t) for t in line.split()]
                    pred_boxes[int(c)].append(xywhn_to_xyxy([x,y,w,h]))
        # 配對
        for cls in range(class_num):
            gts=gt_boxes[cls]; preds=pred_boxes[cls]
            matched=[False]*len(preds)
            for g in gts:
                hit=False
                for idx,p in enumerate(preds):
                    if not matched[idx] and calc_iou(g,p)>=iou_thr:
                        TP[cls]+=1; matched[idx]=True; hit=True; break
                if not hit: FN[cls]+=1
            # 未匹配到的預測為 FP
            FP[cls]+=matched.count(False)
    return TP,FP,FN

# ────────────────── 主程式 ──────────────────
if __name__ == "__main__":
    patch_model = YOLO(PATCH_MODEL_PATH)
    full_model  = YOLO(FULL_MODEL_PATH)

    process_patch(patch_model, INPUT_IMG_DIR, OUTPUT_PATCH_DIR, CLASS_NAMES)
    process_full(full_model , INPUT_IMG_DIR, OUTPUT_FULL_DIR , CLASS_NAMES)

    TP_p,FP_p,FN_p = evaluate(f"{OUTPUT_PATCH_DIR}/labels", GT_LABEL_DIR, len(CLASS_NAMES))
    TP_f,FP_f,FN_f = evaluate(f"{OUTPUT_FULL_DIR}/labels",  GT_LABEL_DIR, len(CLASS_NAMES))

    # ----------- 輸出統計結果 -----------
    print("\n=====  Detection Results vs. Ground‑Truth  =====")
    header = ["Class","GT","Patch TP","Patch Recall","Full TP","Full Recall"]
    rows=[]
    for i,name in enumerate(CLASS_NAMES):
        gt = TP_p[i]+FN_p[i]  # GT 數 = TP+FN (任一模型皆可)
        rec_p = TP_p[i]/gt if gt else 0
        rec_f = TP_f[i]/gt if gt else 0
        rows.append([name, gt, TP_p[i], f"{rec_p:.3f}", TP_f[i], f"{rec_f:.3f}"])
        print(f"{name:20s} | GT:{gt:4d} | Patch TP:{TP_p[i]:4d} R:{rec_p:.3f} | "
              f"Full TP:{TP_f[i]:4d} R:{rec_f:.3f}")
    # 另存 csv
    with open("compare_stats.csv","w",newline="") as csvfile:
        writer = csv.writer(csvfile); writer.writerow(header); writer.writerows(rows)
    print("\n📊 統計已儲存 compare_stats.csv")

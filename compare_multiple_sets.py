import os
import cv2
import random
import shutil
import torch
import csv
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ultralytics import YOLO
from torchvision.ops import nms

# ─────────────────────────────── 基本設定 ───────────────────────────────
PATCH_MODEL_PATH = "patch/weights/best.pt"
FULL_MODEL_PATH  = "full/weights/best.pt"
BASE_INPUT_DIR   = r"D:\Github\RandomPick_v6_6_Full"  # 只需要 train/val
GT_DIR_TEMPLATE  = r"D:\Github\RandomPick_v6_6_Full\{split}\labels"
OUT_PATCH_DIR    = r"D:\Github\CompareResult\patch"
OUT_FULL_DIR     = r"D:\Github\CompareResult\full"
CLASS_NAMES      = ['ship', 'aquaculture cage', 'buoy']
CONF_THRES = 0.5
IOU_THRES = 0.5
CROP_W = CROP_H = 640
VAL_SPLIT = 'val'
SPLITS = ['test1', 'test2', 'test3', 'test4', 'test5']

# ─────────────────────────────── 小工具 ───────────────────────────────
def xywhn_to_xyxy(box):
    x, y, w, h = box
    return [x-w/2, y-h/2, x+w/2, y+h/2]

def calc_iou(box1, box2):
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    if inter == 0: return 0.0
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

def draw_boxes_on_image(img, boxes, colors, names):
    for cls, x, y, w, h, conf in boxes:
        x1, y1 = int((x-w/2)*img.shape[1]), int((y-h/2)*img.shape[0])
        x2, y2 = int((x+w/2)*img.shape[1]), int((y+h/2)*img.shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), colors.get(cls,(255,255,255)), 2)
        cv2.putText(img, f"{names[cls]} {conf:.2f}", (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors.get(cls,(255,255,255)), 1)

def split_val_into_tests(base_dir, seed=42):
    val_img_dir = os.path.join(base_dir, VAL_SPLIT, "images")
    val_lbl_dir = os.path.join(base_dir, VAL_SPLIT, "labels")

    imgs = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png'))]
    random.seed(seed)
    random.shuffle(imgs)
    n = len(imgs) // 5

    for idx in range(5):
        split_name = f"test{idx+1}"
        img_out_dir = os.path.join(base_dir, split_name, "images")
        lbl_out_dir = os.path.join(base_dir, split_name, "labels")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        subset = imgs[idx*n:(idx+1)*n] if idx < 4 else imgs[idx*n:]
        for f in subset:
            shutil.copy(os.path.join(val_img_dir, f), img_out_dir)
            lbl_name = os.path.splitext(f)[0]+'.txt'
            shutil.copy(os.path.join(val_lbl_dir, lbl_name), lbl_out_dir)

# ─────────────────────────────── 推論與評估 ───────────────────────────────
def predict_patch(model, img_dir, out_dir):
    os.makedirs(f"{out_dir}/labels", exist_ok=True)
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    for p in tqdm([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))], desc=f"Patch {os.path.basename(img_dir)}"):
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
        img_draw = img.copy()
        draw_boxes_on_image(img_draw, final, {0:(0,255,0),1:(0,0,255),2:(255,0,0)}, CLASS_NAMES)
        cv2.imwrite(f"{out_dir}/images/{name}.jpg", img_draw)

def predict_full(model, img_dir, out_dir):
    os.makedirs(f"{out_dir}/labels", exist_ok=True)
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    for p in tqdm([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))], desc=f"Full {os.path.basename(img_dir)}"):
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
        img_draw = img.copy()
        draw_boxes_on_image(img_draw, final, {0:(0,255,0),1:(0,0,255),2:(255,0,0)}, CLASS_NAMES)
        cv2.imwrite(f"{out_dir}/images/{name}.jpg", img_draw)

def evaluate(pred_dir, gt_dir, class_num):
    TP=[0]*class_num; FP=[0]*class_num; FN=[0]*class_num
    for lbl in os.listdir(gt_dir):
        if not lbl.endswith('.txt'): continue
        gt_boxes={i:[] for i in range(class_num)}
        with open(os.path.join(gt_dir,lbl)) as f:
            for line in f:
                c,x,y,w,h,*_=[float(t) for t in line.split()]
                gt_boxes[int(c)].append(xywhn_to_xyxy([x,y,w,h]))
        pred_path=os.path.join(pred_dir,lbl)
        pred_boxes={i:[] for i in range(class_num)}
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                for line in f:
                    c,x,y,w,h,*_=[float(t) for t in line.split()]
                    pred_boxes[int(c)].append(xywhn_to_xyxy([x,y,w,h]))
        for cls in range(class_num):
            gts=gt_boxes[cls]; preds=pred_boxes[cls]
            matched=[False]*len(preds)
            for g in gts:
                hit=False
                for idx,p in enumerate(preds):
                    if not matched[idx] and calc_iou(g,p)>=IOU_THRES:
                        TP[cls]+=1; matched[idx]=True; hit=True; break
                if not hit: FN[cls]+=1
            FP[cls]+=matched.count(False)
    return TP,FP,FN

# ─────────────────────────────── 主程式 ───────────────────────────────
if __name__ == "__main__":
    patch_model = YOLO(PATCH_MODEL_PATH)
    full_model  = YOLO(FULL_MODEL_PATH)

    # 自動分 val 成 test1~5（如果 test1/ 不存在）
    if not os.path.exists(os.path.join(BASE_INPUT_DIR, "test1")):
        print("\n🧩 正在從 val 自動分出 test1 ~ test5...")
        split_val_into_tests(BASE_INPUT_DIR)

    os.makedirs(OUT_PATCH_DIR, exist_ok=True)
    os.makedirs(OUT_FULL_DIR, exist_ok=True)

    all_results = []

    for split in SPLITS:
        input_dir = os.path.join(BASE_INPUT_DIR, split, "images")
        gt_dir = GT_DIR_TEMPLATE.format(split=split)

        patch_out_dir = os.path.join(OUT_PATCH_DIR, split)
        full_out_dir = os.path.join(OUT_FULL_DIR, split)

        predict_patch(patch_model, input_dir, patch_out_dir)
        predict_full(full_model, input_dir, full_out_dir)

        TPp, FPp, FNp = evaluate(os.path.join(patch_out_dir, "labels"), gt_dir, len(CLASS_NAMES))
        TPf, FPf, FNf = evaluate(os.path.join(full_out_dir, "labels"), gt_dir, len(CLASS_NAMES))

        all_results.append((split, TPp, FPp, FNp, TPf, FPf, FNf))

    # 統計輸出 CSV
    with open("compare_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Split","Class","GT","Patch TP","Patch Recall","Full TP","Full Recall"])
        for split, TPp, FPp, FNp, TPf, FPf, FNf in all_results:
            for i,name in enumerate(CLASS_NAMES):
                gt = TPp[i] + FNp[i]
                rec_p = TPp[i]/gt if gt else 0
                rec_f = TPf[i]/gt if gt else 0
                writer.writerow([split, name, gt, TPp[i], f"{rec_p:.3f}", TPf[i], f"{rec_f:.3f}"])

    # 畫出 Recall 比較圖
    splits = []
    patch_recall = []
    full_recall = []
    for split, TPp, FPp, FNp, TPf, FPf, FNf in all_results:
        gt_total = sum(TPp) + sum(FNp)
        patch_r = sum(TPp)/gt_total if gt_total else 0
        full_r = sum(TPf)/gt_total if gt_total else 0
        splits.append(split)
        patch_recall.append(patch_r)
        full_recall.append(full_r)

    x = range(len(splits))
    plt.figure(figsize=(10,6))
    plt.bar([i-0.2 for i in x], patch_recall, width=0.4, label='Patch Model Recall', align='center')
    plt.bar([i+0.2 for i in x], full_recall, width=0.4, label='Full Model Recall', align='center')
    plt.xticks(x, splits)
    plt.xlabel('Dataset Split')
    plt.ylabel('Recall')
    plt.title('Patch Model vs Full Model Recall Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('recall_comparison.png')

    print("\n📊 compare_results.csv 已完成！")
    print("📈 recall_comparison.png 已完成！")

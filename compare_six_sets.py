# compare_yolo_models_patchmerge_split5.py  ⟪完整修正版⟫
"""
• 將 val 均分為 test1–test5（固定 seed=42，可自行修改）
• 對每組資料集進行 Patch 與 Full 推論
• 產出：
    ├─ 推論標註  output_patch_<set>/labels/*.txt
    ├─ 推論影像  output_patch_<set>/images/*.jpg   ← ✅ 已加入可視化存圖
    ├─ 同理 output_full_<set>/...
    ├─ compare_sets.csv   （六組資料集，各模式三類召回率）
    └─ recall_sets.png    （平均召回率長條圖）
"""

import random, csv, shutil, os
from pathlib import Path
from typing import List, Dict
import cv2, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.ops import nms

# ───── 基本設定 ─────
PATCH_MODEL_PATH = r"runs_patch\\exp_patch\\weights\\best.pt"
FULL_MODEL_PATH  = r"runs_full\\exp_full\\weights\\best.pt"
IMG_DIR          = Path(r"D:\\Github\\RandomPick_v6_6_Full\\val\\images")
LBL_DIR          = Path(r"D:\\Github\\RandomPick_v6_6_Full\\val\\labels")
WORK_DIR         = Path(r"D:\\Github\\CompareResult")
BASE_DIR         = Path(r"D:\\Github\\RandomPick_v6_6_Full")       # ← 根目錄
CLASS_NAMES      = ['ship', 'aquaculture cage', 'buoy']
CONF_THRES       = 0.5
IOU_THRES        = 0.5
EDGE_THRES       = 5
CROP_W = CROP_H  = 640
SPLIT_SEED       = 42
COLORS           = {0:(0,255,0),1:(0,0,255),2:(255,0,0)}  # BGR

# ───── 公用函式 ─────

def xywhn_to_xyxy(box):
    x,y,w,h=box; return [x-w/2,y-h/2,x+w/2,y+h/2]

def calc_iou(b1,b2):
    xa,ya=max(b1[0],b2[0]),max(b1[1],b2[1]); xb,yb=min(b1[2],b2[2]),min(b1[3],b2[3])
    inter=max(0,xb-xa)*max(0,yb-ya)
    if not inter: return 0.0
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter)

def draw_boxes(img, boxes):
    h,w=img.shape[:2]
    for cls,x,y,bw,bh,conf in boxes:
        x1,y1=int((x-bw/2)*w),int((y-bh/2)*h)
        x2,y2=int((x+bw/2)*w),int((y+bh/2)*h)
        cv2.rectangle(img,(x1,y1),(x2,y2),COLORS.get(cls,(255,255,255)),2)
        cv2.putText(img,f"{CLASS_NAMES[cls]} {conf:.2f}",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,COLORS.get(cls,(255,255,255)),1)

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

# ─────────────────── 推論函式（支援自訂檔名清單） ───────────────────
# ───── 推論函式 ─────

def _infer_patch(model:YOLO,img_dir:Path,out_dir:Path,files:List[str]):
    (out_dir/"labels").mkdir(parents=True,exist_ok=True)
    (out_dir/"images").mkdir(parents=True,exist_ok=True)
    for fn in tqdm(files,desc=f"Patch▶{out_dir.name}",dynamic_ncols=True,leave=False):
        img=cv2.imread(str(img_dir/fn));
        if img is None: continue
        h,w=img.shape[:2]; raw=[]
        for r in range(h//CROP_H):
            for c in range(w//CROP_W):
                x0,y0=c*CROP_W,r*CROP_H
                patch=img[y0:y0+CROP_H,x0:x0+CROP_W]
                for res in model(patch,verbose=False):
                    for b in res.boxes:
                        if b.conf.item()<CONF_THRES: continue
                        cls=int(b.cls);x,y,bw,bh=b.xywh[0].tolist(); raw.append([cls,x0+x,y0+y,bw,bh,b.conf.item()])
        if not raw: continue
        # NMS
        nms_boxes=[]
        for cls in set(b[0] for b in raw):
            cb=[b for b in raw if b[0]==cls]
            xyxy=torch.tensor([[b[1]-b[3]/2,b[2]-b[4]/2,b[1]+b[3]/2,b[2]+b[4]/2] for b in cb])
            confs=torch.tensor([b[5] for b in cb]); keep=nms(xyxy,confs,IOU_THRES)
            nms_boxes.extend([cb[i] for i in keep])
        merged=merge_boxes_across_patches(nms_boxes,w,h,CROP_W,CROP_H,EDGE_THRES)
        final=[[b[0],b[1]/w,b[2]/h,b[3]/w,b[4]/h,b[5]] for b in merged]
        # txt
        (out_dir/"labels"/f"{Path(fn).stem}.txt").write_text("\n".join(
            f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {cf:.4f}" for c,x,y,bw,bh,cf in final),"utf-8")
        # 可視化
        vis=img.copy(); draw_boxes(vis,final); cv2.imwrite(str(out_dir/"images"/fn),vis)

def _infer_full(model:YOLO,img_dir:Path,out_dir:Path,files:List[str]):
    (out_dir/"labels").mkdir(parents=True,exist_ok=True)
    (out_dir/"images").mkdir(parents=True,exist_ok=True)
    for fn in tqdm(files,desc=f"Full ▶{out_dir.name}",dynamic_ncols=True,leave=False):
        img=cv2.imread(str(img_dir/fn));
        if img is None: continue
        h,w=img.shape[:2]; final=[]
        for b in model(img,verbose=False)[0].boxes:
            if b.conf.item()<CONF_THRES: continue
            cls=int(b.cls);x,y,bw,bh=b.xywhn[0].tolist(); final.append([cls,x,y,bw,bh,b.conf.item()])
        # txt
        (out_dir/"labels"/f"{Path(fn).stem}.txt").write_text("\n".join(
            f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f} {cf:.4f}" for c,x,y,bw,bh,cf in final),"utf-8")
        vis=img.copy(); draw_boxes(vis,final); cv2.imwrite(str(out_dir/"images"/fn),vis)

# ───── 評估 ─────

def evaluate(pred_dir:Path,gt_dir:Path,class_num:int):
    TP,FN=[0]*class_num,[0]*class_num
    for gtf in gt_dir.glob("*.txt"):
        if gtf.name=="classes.txt": continue
        gt={i:[] for i in range(class_num)}; pr={i:[] for i in range(class_num)}
        for ln in gtf.read_text().splitlines():
            try:c,x,y,w,h=[float(t) for t in ln.split()[:5]]
            except: continue
            if w<=0 or h<=0: continue
            gt[int(c)].append(xywhn_to_xyxy([x,y,w,h]))
        pf=pred_dir/gtf.name
        if pf.exists():
            for ln in pf.read_text().splitlines():
                try:c,x,y,w,h=[float(t) for t in ln.split()[:5]]
                except: continue
                if w<=0 or h<=0: continue
                pr[int(c)].append(xywhn_to_xyxy([x,y,w,h]))
        for cls in range(class_num):
            match=[False]*len(pr[cls])
            for g in gt[cls]:
                hit=False
                for i,p in enumerate(pr[cls]):
                    if not match[i] and calc_iou(g,p)>=IOU_THRES:
                        TP[cls]+=1; match[i]=True; hit=True; break
                if not hit: FN[cls]+=1
    rec=[TP[c]/(TP[c]+FN[c]) if TP[c]+FN[c] else 0 for c in range(class_num)]
    return rec
# ───── 主程式 ─────
if __name__=="__main__":
    random.seed(SPLIT_SEED)
    patch_model=YOLO(PATCH_MODEL_PATH); full_model=YOLO(FULL_MODEL_PATH)

    img_list=sorted([p.name for p in IMG_DIR.iterdir() if p.suffix.lower() in {'.jpg','.png'}])
    random.shuffle(img_list); k=len(img_list)//5
    splits={f"test{i+1}":img_list[i*k:(i+1)*k] for i in range(5)}; splits["val"]=img_list

    results:Dict[str,Dict]= {}

    for s, flist in splits.items():
        # ——— 複製原圖與 GT 到 test*/
        if s!="val":
            sub_img=(BASE_DIR/s/"images"); sub_lbl=(BASE_DIR/s/"labels")
            sub_img.mkdir(parents=True,exist_ok=True); sub_lbl.mkdir(parents=True,exist_ok=True)
            for fname in flist:
                shutil.copy(IMG_DIR/fname, sub_img/fname)
                lbl_f=LBL_DIR/f"{Path(fname).stem}.txt";  
        # ——— 執行推論與評估
        patch_out=WORK_DIR/f"output_patch_{s}"
        full_out =WORK_DIR/f"output_full_{s}"
        _infer_patch(patch_model,IMG_DIR,patch_out,flist)
        _infer_full(full_model,IMG_DIR,full_out,flist)
        rec_patch = evaluate(patch_out/"labels", LBL_DIR, len(CLASS_NAMES))
        rec_full  = evaluate(full_out/"labels", LBL_DIR, len(CLASS_NAMES))
        results[s]={"patch":rec_patch,"full":rec_full}

    # ——— 寫出 compare_sets.csv
    rows=[["Set","Mode"]+CLASS_NAMES]
    for s in sorted(results.keys(),key=lambda x:(x!="val",x)):
        rows.append([s,"patch"]+[f"{r:.3f}" for r in results[s]["patch"]])
        rows.append([s,"full" ]+[f"{r:.3f}" for r in results[s]["full"]])
    (WORK_DIR/"compare_sets.csv").write_text("\n".join(",".join(r) for r in rows),"utf-8")
    print("✅ compare_sets.csv 已存至",WORK_DIR)

    # ——— 畫圖 recall_sets.png
    sets = sorted(results.keys(),key=lambda x:(x!="val",x))
    patch_avg=[sum(results[s]["patch"])/len(CLASS_NAMES) for s in sets]
    full_avg =[sum(results[s]["full" ])/len(CLASS_NAMES) for s in sets]
    x=range(len(sets)); w=0.35
    plt.figure(figsize=(10,6))
    plt.bar([i-w/2 for i in x],patch_avg,w,label="Patch",alpha=0.7)
    plt.bar([i+w/2 for i in x],full_avg ,w,label="Full" ,alpha=0.7)
    plt.xticks(list(x),sets); plt.ylabel("Average Recall"); plt.title("Patch vs Full Recall")
    plt.grid(axis="y",ls="--",alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(WORK_DIR/"recall_sets.png")
    print("📈 recall_sets.png 已存至",WORK_DIR)

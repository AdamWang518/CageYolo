import os
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import torch

# === 可調整參數 ===
MODEL_PATH = "full\\weights\\best.pt"  # 模型權重
INPUT_FOLDER = "D:\\Github\\RandomPick_v6_5_Combined\\test/images"  # 圖片來源
OUTPUT_FOLDER = "D:\\Github\\CompareResult"  # 輸出資料夾
CLASS_NAMES = ['ship', 'aquaculture cage', 'buoy']  # 類別名稱
CONFIDENCE_THRESHOLD = 0.5

# === 建立輸出資料夾 ===
os.makedirs(os.path.join(OUTPUT_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'labels'), exist_ok=True)

# === 類別對應顏色（可自定義） ===
COLORS = {
    0: (0, 255, 0),      # ship → 綠
    1: (255, 0, 0),      # cage → 藍
    2: (0, 0, 255),      # buoy → 紅
}

# === 載入模型 ===
model = YOLO(MODEL_PATH)

# === 圖片清單 ===
image_paths = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# === 開始推論 ===
for image_path in tqdm(image_paths, desc="YOLO 推論中"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖：{image_path}")
        continue

    h, w = image.shape[:2]
    name = os.path.splitext(os.path.basename(image_path))[0]

    results = model(image, verbose=False)[0]
    boxes = []

    for box in results.boxes:
        cls = int(box.cls)
        x, y, bw, bh = box.xywh[0].tolist()
        conf = box.conf.item()
        if conf >= CONFIDENCE_THRESHOLD:
            boxes.append([cls, x, y, bw, bh, conf])

    # === 畫框與標註 ===
    drawn = image.copy()
    for cls, x, y, bw, bh, conf in boxes:
        x1, y1 = int(x - bw / 2), int(y - bh / 2)
        x2, y2 = int(x + bw / 2), int(y + bh / 2)
        color = COLORS.get(cls, (255, 255, 255))  # fallback: 白
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
        cv2.putText(drawn, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # === 儲存結果圖與標註 ===
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'images', f"{name}.jpg"), drawn)

    with open(os.path.join(OUTPUT_FOLDER, 'labels', f"{name}.txt"), 'w') as f:
        for cls, x, y, bw, bh, conf in boxes:
            f.write(f"{cls} {x/w:.6f} {y/h:.6f} {bw/w:.6f} {bh/h:.6f} {conf:.6f}\n")

print(f"\n✅ 推論完成！已儲存於：{OUTPUT_FOLDER}")

import os
import random
import shutil
from tqdm import tqdm  # ✅ 進度條套件

# === 使用者可自訂的路徑 ===
input_images_dir = "D:\\Github\\RandomPick_v4\\images"
input_labels_dir = "D:\\Github\\RandomPick_v4\\labels"
output_base_dir = "D:\\Github\\RandomPick_v4_Origin"

# 建立輸出資料夾結構
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_base_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, split, "labels"), exist_ok=True)

# 蒐集並打亂圖片檔案
image_files = [f for f in os.listdir(input_images_dir) if f.endswith(".jpg")]
image_files.sort()
random.shuffle(image_files)

# 8:2 分割
split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# 搬移（複製）檔案到新目錄（保留原始檔）
def copy_files(file_list, split):
    print(f"\n📦 開始處理 {split} 資料，共 {len(file_list)} 張圖片")
    for img_file in tqdm(file_list, desc=f"處理 {split}", ncols=80):
        label_file = img_file.replace(".jpg", ".txt")

        src_img = os.path.join(input_images_dir, img_file)
        src_label = os.path.join(input_labels_dir, label_file)

        dst_img = os.path.join(output_base_dir, split, "images", img_file)
        dst_label = os.path.join(output_base_dir, split, "labels", label_file)

        if os.path.exists(src_img) and os.path.exists(src_label):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_label)
        else:
            print(f"[⚠️ 警告] 找不到圖片或標註：{img_file}")

# 執行搬移
copy_files(train_files, "train")
copy_files(val_files, "val")

print("\n✅ 分割完成！")
print("Train:", len(train_files), "張")
print("Val  :", len(val_files), "張")
print(f"➡️ 資料已輸出至：{output_base_dir}\\train 和 {output_base_dir}\\val")

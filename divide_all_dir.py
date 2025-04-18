import os
import cv2
import random
import shutil
from tqdm import tqdm  # ✅ 加入進度條

def split_dataset(image_dir, split_ratio=0.8, seed=42):
    random.seed(seed)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    return image_files[:split_idx], image_files[split_idx:]

def copy_files(file_list, image_dir, label_dir, out_img_dir, out_lbl_dir, desc="Copying"):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    for f in tqdm(file_list, desc=desc, ncols=100):
        img_src = os.path.join(image_dir, f)
        lbl_src = os.path.join(label_dir, os.path.splitext(f)[0] + '.txt')
        if os.path.exists(img_src):
            shutil.copy(img_src, os.path.join(out_img_dir, f))
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(out_lbl_dir, os.path.basename(lbl_src)))

def process_images_and_labels_single(img, image_file, label_path, output_dir, img_width, img_height, crop_width, crop_height):
    patch_all_images_dir = os.path.join(output_dir, "images")
    patch_all_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(patch_all_images_dir, exist_ok=True)
    os.makedirs(patch_all_labels_dir, exist_ok=True)

    if not os.path.exists(label_path):
        print(f"標註文件 {label_path} 不存在，跳過。")
        return

    with open(label_path, 'r') as file:
        annotations = [line.strip().split() for line in file.readlines()]

    cols = img_width // crop_width
    rows = img_height // crop_height

    for i in range(rows):
        for j in range(cols):
            x_start = j * crop_width
            y_start = i * crop_height
            cropped_img = img[y_start:y_start + crop_height, x_start:x_start + crop_width]
            cropped_img_name = f"{os.path.splitext(image_file)[0]}_cropped_{i}_{j}.jpg"
            cropped_img_path = os.path.join(patch_all_images_dir, cropped_img_name)
            cv2.imwrite(cropped_img_path, cropped_img)

            cropped_label_name = f"{os.path.splitext(image_file)[0]}_cropped_{i}_{j}.txt"
            cropped_label_path = os.path.join(patch_all_labels_dir, cropped_label_name)

            with open(cropped_label_path, 'w') as new_file:
                for annotation in annotations:
                    class_id = int(annotation[0])
                    x_center = float(annotation[1]) * img_width
                    y_center = float(annotation[2]) * img_height
                    bbox_width = float(annotation[3]) * img_width
                    bbox_height = float(annotation[4]) * img_height

                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2
                    x_max = x_center + bbox_width / 2
                    y_max = y_center + bbox_height / 2

                    x_min_new = max(0, x_min - x_start)
                    y_min_new = max(0, y_min - y_start)
                    x_max_new = min(crop_width, x_max - x_start)
                    y_max_new = min(crop_height, y_max - y_start)

                    new_bbox_width = x_max_new - x_min_new
                    new_bbox_height = y_max_new - y_min_new

                    if new_bbox_width > 0 and new_bbox_height > 0:
                        new_x_center = (x_min_new + x_max_new) / 2 / crop_width
                        new_y_center = (y_min_new + y_max_new) / 2 / crop_height
                        new_bbox_width /= crop_width
                        new_bbox_height /= crop_height
                        new_file.write(f"{class_id} {new_x_center} {new_y_center} {new_bbox_width} {new_bbox_height}\n")

def export_dual_datasets(image_dir, label_dir, full_output_dir, patch_output_dir,
                         img_width, img_height, crop_width, crop_height):
    train_files, test_files = split_dataset(image_dir)

    # 輸出原圖資料集（full）
    copy_files(train_files, image_dir, label_dir,
               os.path.join(full_output_dir, "train/images"),
               os.path.join(full_output_dir, "train/labels"),
               desc="📦 複製原圖訓練集")
    
    copy_files(test_files, image_dir, label_dir,
               os.path.join(full_output_dir, "test/images"),
               os.path.join(full_output_dir, "test/labels"),
               desc="📦 複製原圖測試集")

    # 輸出 patch 資料集（patch）
    for split_name, files in [("train", train_files), ("test", test_files)]:
        tqdm_iter = tqdm(files, desc=f"🔧 切 patch（{split_name}）", ncols=100)
        for fname in tqdm_iter:
            img_path = os.path.join(image_dir, fname)
            lbl_path = os.path.join(label_dir, os.path.splitext(fname)[0] + '.txt')
            img = cv2.imread(img_path)
            if img is not None:
                process_images_and_labels_single(
                    img, fname, lbl_path,
                    os.path.join(patch_output_dir, split_name),
                    img_width, img_height, crop_width, crop_height
                )

# 使用方式（請依實際修改）
image_dir = 'D:\\Github\\RandomPick_v6_Combined\\images'
label_dir = 'D:\\Github\\RandomPick_v6_Combined\\labels'
full_output_dir = 'D:\\Github\\RandomPick_v6_5_Combined'
patch_output_dir = 'D:\\Github\\RandomPick_v6_5_Patched'

export_dual_datasets(image_dir, label_dir, full_output_dir, patch_output_dir,
                     2560, 1920, 640, 640)

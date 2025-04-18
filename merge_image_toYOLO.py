import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
from colorama import just_fix_windows_console

just_fix_windows_console()

def collect_all_patches(image_root):
    patch_map = {}
    for root, dirs, files in os.walk(image_root):
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

            key = original_name
            if key not in patch_map:
                patch_map[key] = []
            full_path = os.path.join(root, fname)
            patch_map[key].append((row, col, full_path))
    return patch_map

def merge_patches_all(image_root, label_root, output_dir,
                      img_width, img_height, crop_width, crop_height):
    patch_all_images_dir = os.path.join(output_dir, "images")
    patch_all_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(patch_all_images_dir, exist_ok=True)
    os.makedirs(patch_all_labels_dir, exist_ok=True)

    patch_map = collect_all_patches(image_root)
    expected_rows = img_height // crop_height
    expected_cols = img_width // crop_width

    for original_name in tqdm(
        sorted(patch_map.keys()),
        desc="🧩 正在拼接",
        ncols=100,
        ascii=True,           # ✅ 穩定 ASCII 條
        smoothing=0,          # ✅ 不要動畫刷新
        mininterval=0.1,      # ✅ 降低頻繁更新
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout
    ):
        canvas = 255 * np.ones((img_height, img_width, 3), dtype=np.uint8)
        full_annotations = []
        patches = patch_map[original_name]
        patch_positions = {(r, c): path for r, c, path in patches}

        for r in range(expected_rows):
            for c in range(expected_cols):
                if (r, c) not in patch_positions:
                    continue  # 缺 patch，保留白底
                patch_img_path = patch_positions[(r, c)]
                x_offset = c * crop_width
                y_offset = r * crop_height

                patch_img = cv2.imread(patch_img_path)
                if patch_img is None:
                    continue  # 讀不到圖片，保留白底

                canvas[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width] = patch_img

                # 對應標註位置
                patch_label_path = patch_img_path.replace("images", "labels").replace('.jpg', '.txt').replace('.png', '.txt')
                if os.path.exists(patch_label_path):
                    with open(patch_label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            cls, xc, yc, w, h = map(float, parts)
                            abs_xc = (xc * crop_width + x_offset) / img_width
                            abs_yc = (yc * crop_height + y_offset) / img_height
                            abs_w = w * crop_width / img_width
                            abs_h = h * crop_height / img_height
                            full_annotations.append(f"{int(cls)} {abs_xc:.6f} {abs_yc:.6f} {abs_w:.6f} {abs_h:.6f}")

        # 儲存拼接結果
        output_image_path = os.path.join(patch_all_images_dir, original_name + ".jpg")
        output_label_path = os.path.join(patch_all_labels_dir, original_name + ".txt")
        cv2.imwrite(output_image_path, canvas)
        with open(output_label_path, 'w') as f:
            f.write("\n".join(full_annotations))


# === 執行設定 ===
image_root = "D:\\Github\\RandomPick_v6"         # 含所有 patch 的資料夾
label_root = "D:\\Github\\RandomPick_v6"         # 標註也在這個樹狀結構中
output_dir = "D:\\Github\\RandomPick_v6_Combined"  # 輸出還原後資料夾

merge_patches_all(
    image_root=image_root,
    label_root=label_root,
    output_dir=output_dir,
    img_width=2560,
    img_height=1920,
    crop_width=640,
    crop_height=640
)

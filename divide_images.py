"""

"""

import cv2
import os

def process_images_and_labels(image_dir, label_dir, output_dir, img_width, img_height, crop_width, crop_height):
    # 设置输出资料夹
    patch_all_images_dir = os.path.join(output_dir, "images")
    patch_all_labels_dir = os.path.join(output_dir, "labels")

    # 创建资料夹
    os.makedirs(patch_all_images_dir, exist_ok=True)
    os.makedirs(patch_all_labels_dir, exist_ok=True)

    # 获取所有图片文件名
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def process_image(img, image_file, label_path):
        if not os.path.exists(label_path):
            print(f"标注文件 {label_path} 不存在，跳过此图片。")
            return

        # 读取标注文件
        with open(label_path, 'r') as file:
            annotations = [line.strip().split() for line in file.readlines()]

        # 计算切割图片所需的行列数
        cols = img_width // crop_width
        rows = img_height // crop_height

        # 开始切割图片并生成标注
        for i in range(rows):
            for j in range(cols):
                x_start = j * crop_width
                y_start = i * crop_height
                cropped_img = img[y_start:y_start + crop_height, x_start:x_start + crop_width]
                cropped_img_name = f"{os.path.splitext(image_file)[0]}_cropped_{i}_{j}.jpg"
                cropped_img_path = os.path.join(patch_all_images_dir, cropped_img_name)
                cv2.imwrite(cropped_img_path, cropped_img)

                # 创建对应的标注文件
                cropped_label_name = f"{os.path.splitext(image_file)[0]}_cropped_{i}_{j}.txt"
                cropped_label_path = os.path.join(patch_all_labels_dir, cropped_label_name)

                with open(cropped_label_path, 'w') as new_file:
                    for annotation in annotations:
                        class_id = int(annotation[0])
                        x_center = float(annotation[1]) * img_width
                        y_center = float(annotation[2]) * img_height
                        bbox_width = float(annotation[3]) * img_width
                        bbox_height = float(annotation[4]) * img_height

                        # 计算物体在新图中的位置
                        x_min = x_center - bbox_width / 2
                        y_min = y_center - bbox_height / 2
                        x_max = x_center + bbox_width / 2
                        y_max = y_center + bbox_height / 2

                        # 裁剪框的边界与切片图像对齐
                        x_min_new = max(0, x_min - x_start)
                        y_min_new = max(0, y_min - y_start)
                        x_max_new = min(crop_width, x_max - x_start)
                        y_max_new = min(crop_height, y_max - y_start)

                        # 计算新的中心和宽高
                        new_bbox_width = x_max_new - x_min_new
                        new_bbox_height = y_max_new - y_min_new

                        if new_bbox_width > 0 and new_bbox_height > 0:
                            new_x_center = (x_min_new + x_max_new) / 2 / crop_width
                            new_y_center = (y_min_new + y_max_new) / 2 / crop_height
                            new_bbox_width /= crop_width
                            new_bbox_height /= crop_height
                            new_file.write(f"{class_id} {new_x_center} {new_y_center} {new_bbox_width} {new_bbox_height}\n")

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # 读取并处理原图
        img = cv2.imread(image_path)
        if img is not None:
            process_image(img, image_file, label_path)
        else:
            print(f"无法读取图片 {image_path}")

# 使用示例
image_dir = 'path/to/images'  # 替换为您的图片资料夹路径
label_dir = 'path/to/labels'  # 替换为您的标注资料夹路径
output_dir = 'path/to/output_dir'  # 输出到 output_dir 资料夹
os.makedirs(output_dir, exist_ok=True)

# 假设图片尺寸为 2560x1920，切片大小为 640x640
process_images_and_labels(image_dir, label_dir, output_dir, 2560, 1920, 640, 640)

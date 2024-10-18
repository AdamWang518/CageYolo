import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import shutil

# 判断两个框是否不重叠且边界长度相似且距离很近
def should_merge_boxes(box1, box2, length_threshold=0.2, distance_threshold=50):
    # box1, box2格式：[cls, x_center, y_center, width, height, confidence]
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    # 判断是否不重叠
    if (x1_max < x2_min or x2_max < x1_min) and (y1_max < y2_min or y2_max < y1_min):
        width1, height1 = box1[3], box1[4]
        width2, height2 = box2[3], box2[4]

        # 如果宽度相似，判断它们在垂直方向的距离是否小于阈值
        if abs(width1 - width2) / min(width1, width2) < length_threshold:
            vertical_distance = min(abs(y1_min - y2_max), abs(y2_min - y1_max))
            if vertical_distance < distance_threshold:
                return True

        # 如果高度相似，判断它们在水平方向的距离是否小于阈值
        if abs(height1 - height2) / min(height1, height2) < length_threshold:
            horizontal_distance = min(abs(x1_min - x2_max), abs(x2_min - x1_max))
            if horizontal_distance < distance_threshold:
                return True

    return False

# 合并相近或接近条件的框
def merge_nearby_boxes_by_edge(boxes, length_threshold=0.2, distance_threshold=50):
    merged_boxes = []
    used_boxes = set()

    for i, box1 in enumerate(boxes):
        if i in used_boxes:
            continue

        cls1 = box1[0]
        if cls1 == 2:  # 浮标类别，不合并
            merged_boxes.append(box1)
            used_boxes.add(i)
            continue

        merged_group = [box1]
        confidences = [box1[5]]  # 收集置信度

        for j, box2 in enumerate(boxes):
            if i == j or j in used_boxes:
                continue

            cls2 = box2[0]
            if cls1 == cls2 and should_merge_boxes(box1, box2, length_threshold, distance_threshold):
                merged_group.append(box2)
                confidences.append(box2[5])
                used_boxes.add(j)

        if len(merged_group) > 1:
            x_min = min([box[1] - box[3] / 2 for box in merged_group])
            y_min = min([box[2] - box[4] / 2 for box in merged_group])
            x_max = max([box[1] + box[3] / 2 for box in merged_group])
            y_max = max([box[2] + box[4] / 2 for box in merged_group])

            avg_x_center = (x_min + x_max) / 2
            avg_y_center = (y_min + y_max) / 2
            avg_width = x_max - x_min
            avg_height = y_max - y_min
            min_confidence = min(confidences)  # 合并后的置信度取最小值

            merged_boxes.append([cls1, avg_x_center, avg_y_center, avg_width, avg_height, min_confidence])
        else:
            merged_boxes.append(box1)

        used_boxes.add(i)

    return merged_boxes

# 绘制物体框在图像上
def draw_boxes_on_image(image, boxes, class_names, line_thickness=1, font_scale=0.5):
    colors = {
        0: (0, 255, 0),  # 绿色 for class 0
        1: (255, 0, 0),  # 蓝色 for class 1
        2: (0, 0, 255)   # 红色 for class 2
    }

    for box in boxes:
        cls = int(box[0])
        x_center = box[1]
        y_center = box[2]
        width = box[3]
        height = box[4]
        confidence = box[5]

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        color = colors.get(cls, (0, 255, 0))

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, line_thickness)

        label = f"{confidence:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        text_x = x_min
        text_y = y_min - 5 if y_min - 5 > 10 else y_min + 15
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

# 处理并预测单张图片
def process_and_predict(image_path, output_dirs, class_names, confidence_threshold=0.0, crop_width=640, crop_height=640, length_threshold=0.2, distance_threshold=50):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片 {image_path}")
        return

    img_height, img_width, _ = image.shape

    image_name = os.path.basename(image_path).split('.')[0]

    cols = img_width // crop_width
    rows = img_height // crop_height

    all_boxes = []

    for i in range(rows):
        for j in range(cols):
            x_start = j * crop_width
            y_start = i * crop_height
            cropped_img = image[y_start:y_start + crop_height, x_start:x_start + crop_width]

            results = model(cropped_img, verbose=False)

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    x_center = box.xywh[0][0].item() + x_start
                    y_center = box.xywh[0][1].item() + y_start
                    width = box.xywh[0][2].item()
                    height = box.xywh[0][3].item()
                    confidence = box.conf.item()

                    all_boxes.append([cls, x_center, y_center, width, height, confidence])

    # 保存所有检测到的物体
    image_all = image.copy()
    draw_boxes_on_image(image_all, all_boxes, class_names)
    cv2.imwrite(os.path.join(output_dirs['all_objects'], f"{image_name}.jpg"), image_all)

    # 根据置信度阈值筛选物体
    thresholded_boxes = [box for box in all_boxes if box[5] >= confidence_threshold]

    # 保存经过阈值筛选的物体
    image_thresholded = image.copy()
    draw_boxes_on_image(image_thresholded, thresholded_boxes, class_names)
    cv2.imwrite(os.path.join(output_dirs['thresholded_objects'], f"{image_name}.jpg"), image_thresholded)

    # 合并物体框
    merged_boxes = merge_nearby_boxes_by_edge(thresholded_boxes, length_threshold, distance_threshold)

    # 保存合并后的物体
    image_merged = image.copy()
    draw_boxes_on_image(image_merged, merged_boxes, class_names)
    cv2.imwrite(os.path.join(output_dirs['merged_objects'], f"{image_name}.jpg"), image_merged)

    # 保存 YOLO 格式的标注
    save_yolo_format(merged_boxes, image_name, output_dirs['labels'], img_width, img_height)

# 保存 YOLO 格式标注
def save_yolo_format(boxes, image_name, save_dir, img_width, img_height):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    txt_file_path = os.path.join(save_dir, f"{image_name}.txt")

    with open(txt_file_path, "w") as f:
        for box in boxes:
            cls = int(box[0])
            x_center = box[1] / img_width
            y_center = box[2] / img_height
            width = box[3] / img_width
            height = box[4] / img_height
            confidence = box[5]

            f.write(f"{cls} {x_center} {y_center} {width} {height} {confidence}\n")

# 处理整个文件夹的函数
def process_images(input_folder, output_root, class_names, confidence_threshold=0.0, length_threshold=0.2, distance_threshold=50):
    image_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]

    # 创建输出文件夹
    output_dirs = {
        'all_objects': os.path.join(output_root, 'all_objects'),
        'thresholded_objects': os.path.join(output_root, 'thresholded_objects'),
        'merged_objects': os.path.join(output_root, 'merged_objects'),
        'labels': os.path.join(output_root, 'labels')
    }

    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    with tqdm(total=len(image_paths), desc="Processing images", unit="image") as pbar:
        for image_path in image_paths:
            process_and_predict(image_path, output_dirs, class_names, confidence_threshold, length_threshold=length_threshold, distance_threshold=distance_threshold)
            pbar.update(1)

    # 复制 classes.txt 到 labels 文件夹
    shutil.copy('classes.txt', output_dirs['labels'])

# 加载模型
model_path = 'weights/best.engine'
model = YOLO(model_path)

# 输入和输出路径
input_folder = "datasets/predict/0925/images"
output_root = "datasets/predict/0925/output_threshold_05"

# 类别名称
class_names = ['ship', 'aquaculture cage', 'buoy']  # 替换为您的类别名称列表

# 置信度阈值
confidence_threshold = 0.5  # 根据需要设置阈值

# 处理图片
process_images(input_folder, output_root, class_names, confidence_threshold=confidence_threshold, length_threshold=0.2, distance_threshold=50)

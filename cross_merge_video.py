import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import shutil
import torch
from torchvision.ops import nms

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

# 判断框是否靠近 patch 的边缘
def is_near_patch_edge(box, img_width, img_height, crop_width, crop_height, edge_threshold=20):
    """判断框是否靠近 patch 的边缘"""
    x_center, y_center, width, height = box[1], box[2], box[3], box[4]

    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    # 获取框所在的 patch 的位置
    patch_col = int(x_center // crop_width)
    patch_row = int(y_center // crop_height)

    # 计算该 patch 的边界
    patch_x_min = patch_col * crop_width
    patch_y_min = patch_row * crop_height
    patch_x_max = patch_x_min + crop_width
    patch_y_max = patch_y_min + crop_height

    # 判断框是否靠近 patch 的边缘
    near_left = (x_min - patch_x_min) < edge_threshold
    near_right = (patch_x_max - x_max) < edge_threshold
    near_top = (y_min - patch_y_min) < edge_threshold
    near_bottom = (patch_y_max - y_max) < edge_threshold

    return near_left or near_right or near_top or near_bottom

# 判断两个框是否相邻
def boxes_are_adjacent(box1, box2, max_distance=20):
    """判断两个框是否在空间上相邻"""
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    # 判断是否接近（边缘接触或有少量间隙）
    horizontal_adjacent = (abs(x1_min - x2_max) < max_distance or abs(x1_max - x2_min) < max_distance)
    vertical_overlap = not (y1_max < y2_min or y2_max < y1_min)

    vertical_adjacent = (abs(y1_min - y2_max) < max_distance or abs(y1_max - y2_min) < max_distance)
    horizontal_overlap = not (x1_max < x2_min or x2_max < x1_min)

    return (horizontal_adjacent and vertical_overlap) or (vertical_adjacent and horizontal_overlap)

# 合并两个框
def merge_two_boxes(box1, box2):
    """合并两个框"""
    cls = box1[0]
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    x_min = min(x1_min, x2_min)
    y_min = min(y1_min, y2_min)
    x_max = max(x1_max, x2_max)
    y_max = max(y1_max, y2_max)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    confidence = min(box1[5], box2[5])

    return [cls, x_center, y_center, width, height, confidence]

# 合并跨越 patch 的框
def merge_boxes_across_patches(boxes, img_width, img_height, crop_width, crop_height, edge_threshold=20):
    """只合并跨越 patch 的同一物体的框"""
    merged_boxes = []
    used_indices = set()

    # 建立一个从位置到框索引的映射，方便查找相邻 patch 的框
    position_to_indices = {}
    for idx, box in enumerate(boxes):
        x_center, y_center = box[1], box[2]
        patch_col = int(x_center // crop_width)
        patch_row = int(y_center // crop_height)
        key = (patch_row, patch_col)
        position_to_indices.setdefault(key, []).append(idx)

    for idx, box in enumerate(boxes):
        if idx in used_indices:
            continue

        cls = box[0]
        merged_box = box.copy()

        if is_near_patch_edge(box, img_width, img_height, crop_width, crop_height, edge_threshold):
            x_center, y_center = box[1], box[2]
            patch_col = int(x_center // crop_width)
            patch_row = int(y_center // crop_height)

            # 查找相邻的四个方向的 patch
            neighbor_patches = [
                (patch_row - 1, patch_col),     # 上
                (patch_row + 1, patch_col),     # 下
                (patch_row, patch_col - 1),     # 左
                (patch_row, patch_col + 1)      # 右
            ]

            for neighbor in neighbor_patches:
                neighbor_indices = position_to_indices.get(neighbor, [])
                for n_idx in neighbor_indices:
                    if n_idx in used_indices:
                        continue
                    neighbor_box = boxes[n_idx]
                    if neighbor_box[0] != cls:
                        continue
                    # 判断框是否在空间上相邻（边缘重合或接近）
                    if boxes_are_adjacent(merged_box, neighbor_box, max_distance=edge_threshold):
                        # 合并框
                        merged_box = merge_two_boxes(merged_box, neighbor_box)
                        used_indices.add(n_idx)

        merged_boxes.append(merged_box)
        used_indices.add(idx)

    return merged_boxes

# 处理并预测单帧图像
def process_and_predict(image, class_names, confidence_threshold=0.0, crop_width=640, crop_height=640, iou_threshold=0.5, edge_threshold=20):
    img_height, img_width, _ = image.shape
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

    # 根据置信度阈值筛选物体
    thresholded_boxes = [box for box in all_boxes if box[5] >= confidence_threshold]

    if not thresholded_boxes:
        return image

    # 第一步：对每个类别分别进行 NMS，合并重叠的框（重复检测）
    nms_boxes = []
    for cls in set([box[0] for box in thresholded_boxes]):
        cls_boxes = [box for box in thresholded_boxes if box[0] == cls]
        boxes_tensor = torch.tensor([[box[1] - box[3] / 2, box[2] - box[4] / 2,
                                      box[1] + box[3] / 2, box[2] + box[4] / 2] for box in cls_boxes])
        scores_tensor = torch.tensor([box[5] for box in cls_boxes])
        indices = nms(boxes_tensor, scores_tensor, iou_threshold)
        nms_boxes.extend([cls_boxes[i] for i in indices])

    # 第二步：合并跨越 patch 的同一物体的框
    merged_boxes = merge_boxes_across_patches(nms_boxes, img_width, img_height, crop_width, crop_height, edge_threshold)

    # 绘制检测框
    image_with_boxes = image.copy()
    draw_boxes_on_image(image_with_boxes, merged_boxes, class_names)

    return image_with_boxes

# 处理视频的函数
def process_video(input_video_path, output_video_path, class_names, confidence_threshold=0.0, iou_threshold=0.5, edge_threshold=20):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"无法读取视频 {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建视频写入对象
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理单帧图像
            processed_frame = process_and_predict(frame, class_names, confidence_threshold, iou_threshold=iou_threshold, edge_threshold=edge_threshold)

            # 将处理后的帧写入输出视频
            out.write(processed_frame)

            pbar.update(1)

    cap.release()
    out.release()
    print(f"视频处理完成，已保存到 {output_video_path}")

# 加载模型
model_path = 'weights/best.pt'
model = YOLO(model_path)

# 类别名称
class_names = ['ship', 'aquaculture cage', 'buoy']  # 替换为您的类别名称列表

# 置信度阈值
confidence_threshold = 0.5  # 根据需要设置阈值

# IoU 阈值（用于 NMS）
iou_threshold = 0.5  # 通常设置为 0.5

# 边缘阈值（用于判断是否靠近 patch 边缘）
edge_threshold = 5  # 根据实际情况调整

# 输入和输出视频路径
input_video_path = "D:\\Github\\CageYolo\\Test\\input_video.mp4"
output_video_path = "D:\\Github\\CageYolo\\Test\\output_video.mp4"

# 处理视频
process_video(input_video_path, output_video_path, class_names, confidence_threshold=confidence_threshold,
              iou_threshold=iou_threshold, edge_threshold=edge_threshold)

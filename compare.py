import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import torch
from torchvision.ops import nms

def draw_boxes_on_image(image, boxes, class_names, line_thickness=1, font_scale=0.5):
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
    for box in boxes:
        cls, x, y, w, h, conf = box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), colors.get(cls, (0, 255, 0)), line_thickness)
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors.get(cls, (0, 255, 0)), 1)

def save_yolo_format(boxes, image_name, save_dir, w, h):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{image_name}.txt"), 'w') as f:
        for cls, x, y, bw, bh, conf in boxes:
            f.write(f"{cls} {x/w} {y/h} {bw/w} {bh/h} {conf}\n")

def process_images_with_full_model(model, input_folder, output_root, class_names, confidence_threshold=0.5):
    os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'labels'), exist_ok=True)
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
    for image_path in tqdm(image_paths, desc="Full model"):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        results = model(image, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            cls = int(box.cls)
            x, y, bw, bh = box.xywh[0].tolist()
            conf = box.conf.item()
            if conf >= confidence_threshold:
                boxes.append([cls, x, y, bw, bh, conf])
        draw = image.copy()
        draw_boxes_on_image(draw, boxes, class_names)
        cv2.imwrite(os.path.join(output_root, 'images', f"{image_name}.jpg"), draw)
        save_yolo_format(boxes, image_name, os.path.join(output_root, 'labels'), w, h)

def merge_two_boxes(box1, box2):
    cls = box1[0]
    x1_min = box1[1] - box1[3]/2
    y1_min = box1[2] - box1[4]/2
    x1_max = box1[1] + box1[3]/2
    y1_max = box1[2] + box1[4]/2
    x2_min = box2[1] - box2[3]/2
    y2_min = box2[2] - box2[4]/2
    x2_max = box2[1] + box2[3]/2
    y2_max = box2[2] + box2[4]/2
    x_min, y_min = min(x1_min, x2_min), min(y1_min, y2_min)
    x_max, y_max = max(x1_max, x2_max), max(y1_max, y2_max)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    conf = min(box1[5], box2[5])
    return [cls, x_center, y_center, w, h, conf]

def process_images_with_patch_model(model, input_folder, output_root, class_names, confidence_threshold=0.5, crop_width=640, crop_height=640, iou_threshold=0.5, edge_threshold=20):
    dirs = {k: os.path.join(output_root, k) for k in ['merged', 'labels']}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    for path in tqdm(image_paths, desc="Patch model"):
        img = cv2.imread(path)
        if img is None:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        h, w = img.shape[:2]
        boxes = []
        for i in range(h // crop_height):
            for j in range(w // crop_width):
                x0, y0 = j * crop_width, i * crop_height
                patch = img[y0:y0+crop_height, x0:x0+crop_width]
                results = model(patch, verbose=False)
                for r in results:
                    for b in r.boxes:
                        cls = int(b.cls)
                        x = b.xywh[0][0].item() + x0
                        y = b.xywh[0][1].item() + y0
                        bw = b.xywh[0][2].item()
                        bh = b.xywh[0][3].item()
                        conf = b.conf.item()
                        if conf >= confidence_threshold:
                            boxes.append([cls, x, y, bw, bh, conf])
        # NMS by class
        merged = []
        used = set()
        for cls in set(b[0] for b in boxes):
            cls_boxes = [b for b in boxes if b[0] == cls]
            xyxy = torch.tensor([[b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2] for b in cls_boxes])
            confs = torch.tensor([b[5] for b in cls_boxes])
            keep = nms(xyxy, confs, iou_threshold)
            for idx in keep:
                merged.append(cls_boxes[idx])
        draw = img.copy()
        draw_boxes_on_image(draw, merged, class_names)
        cv2.imwrite(os.path.join(dirs['merged'], f"{name}.jpg"), draw)
        save_yolo_format(merged, name, dirs['labels'], w, h)

if __name__ == "__main__":
    patch_model = YOLO("weights/patch/best.pt")
    full_model = YOLO("train4\\weights")
    input_folder = "D:/Github/RandomPick_v6_5_Combined/test/images"
    class_names = ['ship', 'aquaculture cage', 'buoy']

    process_images_with_patch_model(
        model=patch_model,
        input_folder=input_folder,
        output_root="output_patch",
        class_names=class_names,
        confidence_threshold=0.5
    )

    process_images_with_full_model(
        model=full_model,
        input_folder=input_folder,
        output_root="output_full",
        class_names=class_names,
        confidence_threshold=0.5
    )

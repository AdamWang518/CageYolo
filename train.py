from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    results = model.train(
    data="dataset.yaml",
    epochs=36,
    imgsz=640,
    hsv_h=0.05,
    hsv_s=0.8,
    hsv_v=0.6
)


if __name__ == "__main__":
    main()

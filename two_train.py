from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    # 訓練 Full
    results_full = model.train(
        data="dataset_full.yaml",
        epochs=36,
        imgsz=640,
        hsv_h=0.05,
        hsv_s=0.8,
        hsv_v=0.6,
        project="runs_full",    # ⬅️ 自訂主資料夾
        name="exp_full"         # ⬅️ 自訂子資料夾
    )
    # 訓練 Patch
    results_patch = model.train(
        data="dataset_patch.yaml",
        epochs=36,
        imgsz=640,
        hsv_h=0.05,
        hsv_s=0.8,
        hsv_v=0.6,
        project="runs_patch",   # ⬅️ 自訂主資料夾
        name="exp_patch"        # ⬅️ 自訂子資料夾
    )

if __name__ == "__main__":
    main()

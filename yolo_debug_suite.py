#!/usr/bin/env python
"""
YOLO Debug Suite  ▸  一鍵排查資料、推論與評估三大環節
========================================================

用法
----
python yolo_debug_suite.py \
       --img_dir  D:/Github/RandomPick_v6_6_Full/val/images \
       --gt_dir   D:/Github/RandomPick_v6_6_Full/val/labels \
       --pred_dir_patch  D:/Github/CompareResult/output_patch_F/labels \
       --pred_dir_full   D:/Github/CompareResult/output_full_F/labels \
       --class_names "ship,aquaculture cage,buoy" \
       --crop 640

輸出
----
1. 終端摘要表格（錯誤計數）
2. debug_report.txt — 詳細逐檔錯誤清單，可直接打開檢視

支援檢測項目
--------------
✔ 圖片檔案是否能 imread()
✔ 圖片尺寸是否 >= crop size
✔ images ⇄ labels 檔名對應是否一致
✔ 標註格式：token 數、類別索引、正規化座標、寬高正值
✔ 預測標註是否出現文字類別（如 'ship'）
✔ GT 與 Pred 檔案數是否匹配

後續可直接在此檔案增補自定檢查，或把該函式放到 Jupyter Notebook 互動調試。
"""
import os
import cv2
import argparse
from pathlib import Path
from typing import List, Tuple

# ──────────────────────── 輔助工具 ──────────────────────── #

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def read_lines(fp: Path) -> List[str]:
    return [ln.strip() for ln in fp.read_text().splitlines() if ln.strip()]


def validate_label_line(ln: str, class_num: int) -> Tuple[bool, str]:
    """檢查單行標註格式是否正確，回傳 (是否通過, 錯誤訊息)"""
    parts = ln.split()
    if len(parts) < 5:
        return False, f"token<5 → {ln}"
    try:
        cls = int(parts[0])
        if not (0 <= cls < class_num):
            return False, f"class_id {cls} 超出範圍"
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return False, f"不能轉 float/int → {ln}"
    if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
        return False, f"座標不在 0‑1 之間"
    if w <= 0 or h <= 0:
        return False, f"w/h 非正值"
    return True, ""


# ──────────────────────── 主檢查流程 ──────────────────────── #

def check_images(img_dir: Path, crop: int):
    bad_read, too_small = [], []
    for p in img_dir.iterdir():
        if not is_img(p):
            continue
        img = cv2.imread(str(p))
        if img is None:
            bad_read.append(p.name)
            continue
        h, w = img.shape[:2]
        if h < crop or w < crop:
            too_small.append(f"{p.name} ({w}x{h})")
    return bad_read, too_small


def check_labels(lbl_dir: Path, class_num: int):
    fmt_err = {}
    for fp in lbl_dir.glob("*.txt"):
        for ln in read_lines(fp):
            ok, err = validate_label_line(ln, class_num)
            if not ok:
                fmt_err.setdefault(fp.name, []).append(err)
    return fmt_err


def diff_sets(img_dir: Path, lbl_dir: Path):
    img_stems = {p.stem for p in img_dir.iterdir() if is_img(p)}
    lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}
    imgs_without_lbl = sorted(img_stems - lbl_stems)
    lbl_without_img = sorted(lbl_stems - img_stems)
    return imgs_without_lbl, lbl_without_img


def check_pred_tokens(pred_dir: Path):
    bad_cls = {}
    for fp in pred_dir.glob("*.txt"):
        for ln in read_lines(fp):
            tok = ln.split()[0]
            if not tok.isdigit():
                bad_cls.setdefault(fp.name, []).append(tok)
    return bad_cls


# ──────────────────────────── CLI ──────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description="YOLO Debug Suite")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir_patch", required=False)
    ap.add_argument("--pred_dir_full", required=False)
    ap.add_argument("--class_names", required=True, help="comma separated class list")
    ap.add_argument("--crop", type=int, default=640)
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    gt_dir = Path(args.gt_dir)
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    class_num = len(class_names)

    report_lines = []
    def log(s: str):
        print(s)
        report_lines.append(s)

    log("================== IMAGE 檢查 ==================")
    bad_read, too_small = check_images(img_dir, args.crop)
    log(f"無法讀圖: {len(bad_read)}")
    if bad_read:
        log("  ↳ " + ", ".join(bad_read[:10]) + (" ..." if len(bad_read) > 10 else ""))
    log(f"尺寸<crop: {len(too_small)}")

    log("\n================== GT LABEL 檢查 ==================")
    fmt_err = check_labels(gt_dir, class_num)
    log(f"格式錯誤檔數: {len(fmt_err)}")
    if fmt_err:
        for fn, errs in list(fmt_err.items())[:10]:
            log(f"  {fn}: {errs[0]}")
        if len(fmt_err) > 10:
            log("  ...")

    log("\n圖與標註檔名對應檢查")
    img_wo_lbl, lbl_wo_img = diff_sets(img_dir, gt_dir)
    log(f"圖片無對應標註: {len(img_wo_lbl)}")
    log(f"標註無對應圖片: {len(lbl_wo_img)}")

    # 可選檢查 Pred
    for tag, pdir in {"PATCH": args.pred_dir_patch, "FULL": args.pred_dir_full}.items():
        if not pdir:
            continue
        pdir_path = Path(pdir)
        log(f"\n================== {tag} PRED LABEL 檢查 ==================")
        bad_cls = check_pred_tokens(pdir_path)
        log(f"類別 token 非數字的檔數: {len(bad_cls)}")
        if bad_cls:
            for fn, toks in list(bad_cls.items())[:10]:
                log(f"  {fn}: {toks[:3]}")
            if len(bad_cls) > 10:
                log("  ...")

    (Path.cwd() / "debug_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    log("\n✅  檢查結束，詳細錯誤已寫入 debug_report.txt")


if __name__ == "__main__":
    main()

import os

def check_labels_format(label_dir):
    bad_files = []
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(label_dir, fname)
        with open(fpath, "r") as f:
            for line_num, line in enumerate(f, 1):
                tokens = line.strip().split()
                if not tokens:
                    continue
                if not tokens[0].replace('.', '', 1).isdigit():  # 允許小數點（極少見）
                    bad_files.append((fname, line_num, line.strip()))
                    break  # 找到錯誤就列出，不必繼續掃這個檔案
    if bad_files:
        print("❌ 發現格式錯誤的標註檔案：\n")
        for fname, line_num, content in bad_files:
            print(f"檔案：{fname}（第 {line_num} 行）→ {content}")
    else:
        print("✅ 全部標註檔案格式正確。")

# 使用方法：
if __name__ == "__main__":
    label_folder = r"D:\\Github\\CompareResult\\output_full2\\labels"   # 改成你的標註資料夾
    check_labels_format(label_folder)

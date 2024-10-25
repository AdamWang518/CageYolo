import cv2
import os

def images_to_video(image_folder, output_video_path, fps=30):
    # 获取文件夹中的所有图像文件路径
    image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 确保图像文件按名称排序
    image_files.sort()

    # 读取第一张图片，获取其宽度和高度
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐一读取图像并写入视频文件
    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is None:
            print(f"无法读取图片 {image_file}, 跳过...")
            continue
        out.write(img)

    # 释放视频写入对象
    out.release()
    print(f"视频已保存到 {output_video_path}")

# 输入图片文件夹路径和输出视频路径
image_folder = "D:\\Github\\CageYolo\\Test\\0811"
output_video_path = "D:\\Github\\CageYolo\\Test\\0811_origin.mp4"

# 将图片合并为视频
images_to_video(image_folder, output_video_path, fps=30)

"""
使用SAM模型生成整张图片的掩码
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# 配置参数
MODEL_TYPE = "vit_h"  # 模型类型：vit_h, vit_l, vit_b
CHECKPOINT_PATH = "/root/autodl-tmp/fc-clip/segment-anything/sam_vit_h_4b8939.pth"  # 模型权重路径
INPUT_DIR = "/root/autodl-tmp/fc-clip/vol/fishyscapes/LostAndFound/original"  # 输入图片目录
OUTPUT_DIR = "/root/autodl-tmp/fc-clip/segment-anything/output"  # 输出目录

def show_mask(mask, ax, random_color=False):
    """
    可视化掩码
    @param mask: 掩码数据
    @param ax: matplotlib轴对象
    @param random_color: 是否使用随机颜色
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_masks(image, masks, output_path):
    """
    保存掩码可视化结果
    @param image: 原始图片
    @param masks: 生成的掩码列表
    @param output_path: 输出路径
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask['segmentation'], plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

    # 加载SAM模型
    print("正在加载SAM模型...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("模型加载完成")

    # 获取所有图片文件
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 处理每张图片
    for image_file in image_files:
        print(f"正在处理: {image_file}")
        
        # 读取图片
        image_path = os.path.join(INPUT_DIR, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 生成掩码
        masks = mask_generator.generate(image)

        # 保存结果
        base_name = os.path.splitext(image_file)[0]
        
        # 保存原始图片
        cv2.imwrite(os.path.join(OUTPUT_DIR, "images", image_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # 保存掩码数据
        np.save(os.path.join(OUTPUT_DIR, "masks", f"{base_name}_masks.npy"), masks)
        
        # 保存可视化结果
        save_masks(image, masks, os.path.join(OUTPUT_DIR, "visualizations", f"{base_name}_vis.png"))

    print("所有图片处理完成")

if __name__ == "__main__":
    main()
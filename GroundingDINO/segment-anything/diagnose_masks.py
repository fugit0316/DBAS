"""
诊断掩码图像和原始图像的问题
"""

import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

# 配置路径
BINARY_MASKS_DIR = "/root/autodl-tmp/fc-clip/val-mid/binary_masks"
ORIGINAL_IMAGES_DIR = "/root/autodl-tmp/fc-clip/vol/road_anomaly/original"
DIAGNOSIS_DIR = "/root/autodl-tmp/fc-clip/segment-anything/diagnosis"

def get_region_info(region):
    """
    安全地获取区域信息
    """
    area = region.area
    try:
        if region.axis_minor_length > 0:
            aspect_ratio = region.axis_major_length / region.axis_minor_length
        else:
            aspect_ratio = float('inf')  # 如果短轴为0，设置长宽比为无穷大
    except Exception as e:
        aspect_ratio = -1  # 如果计算出错，设置为-1表示无效值
    
    return {
        'area': area,
        'aspect_ratio': aspect_ratio,
        'bbox': region.bbox,
        'centroid': region.centroid
    }

def analyze_mask(mask_path):
    """分析掩码图像"""
    # 读取掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return f"无法读取掩码图像: {mask_path}"

    # 获取图像下半部分
    h, w = mask.shape
    lower_half = mask[h//2:, :]
    
    # 统计信息
    white_pixels = np.sum(lower_half == 255)
    total_pixels = lower_half.size
    white_percentage = (white_pixels / total_pixels) * 100

    # 连通区域分析
    binary = (lower_half == 255).astype(np.uint8)
    labeled = label(binary)
    regions = regionprops(labeled)

    # 收集区域信息
    regions_info = [get_region_info(r) for r in regions]

    # 创建诊断可视化
    plt.figure(figsize=(15, 10))
    
    # 原始掩码
    plt.subplot(221)
    plt.imshow(mask, cmap='gray')
    plt.title('Complete Mask')
    
    # 下半部分
    plt.subplot(222)
    plt.imshow(lower_half, cmap='gray')
    plt.title('Lower Half')
    
    # 连通区域
    plt.subplot(223)
    plt.imshow(labeled, cmap='nipy_spectral')
    plt.title('Connected Components')
    
    # 区域标记
    plt.subplot(224)
    plt.imshow(lower_half, cmap='gray')
    for i, info in enumerate(regions_info):
        y, x = info['centroid']
        plt.plot(x, y, 'r*')
        plt.text(x, y, f"{i+1}", color='red')
    plt.title('Region Centers')

    # 保存诊断图
    os.makedirs(DIAGNOSIS_DIR, exist_ok=True)
    plt.savefig(os.path.join(DIAGNOSIS_DIR, f"diagnosis_{os.path.basename(mask_path)}"))
    plt.close()

    return {
        'white_percentage': white_percentage,
        'region_count': len(regions),
        'regions_info': regions_info
    }

def check_original_images():
    """检查原始图像目录的文件名"""
    if not os.path.exists(ORIGINAL_IMAGES_DIR):
        print(f"警告：原始图像目录不存在: {ORIGINAL_IMAGES_DIR}")
        return set()
    return set(os.listdir(ORIGINAL_IMAGES_DIR))

def main():
    # 获取所有掩码文件
    if not os.path.exists(BINARY_MASKS_DIR):
        print(f"错误：掩码目录不存在: {BINARY_MASKS_DIR}")
        return

    mask_files = [f for f in os.listdir(BINARY_MASKS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    original_files = check_original_images()

    print("=== 文件匹配分析 ===")
    for mask_file in mask_files:
        base_name = os.path.splitext(mask_file)[0]
        found = False
        for ext in ['.png', '.jpg', '.jpeg']:
            if base_name + ext in original_files:
                found = True
                break
        if not found:
            print(f"\n文件名不匹配: {mask_file}")
            print("最相似的原始文件名:")
            # 显示最相似的文件名
            similar_files = [f for f in original_files if any(word.lower() in f.lower() for word in base_name.split('_'))]
            for orig_file in similar_files[:5]:  # 只显示前5个最相似的
                print(f"  - {orig_file}")

    print("\n=== 掩码分析 ===")
    for mask_file in mask_files:
        print(f"\n分析掩码: {mask_file}")
        mask_path = os.path.join(BINARY_MASKS_DIR, mask_file)
        results = analyze_mask(mask_path)
        
        if isinstance(results, str):
            print(results)
            continue
            
        print(f"白色像素占比: {results['white_percentage']:.2f}%")
        print(f"检测到的区域数量: {results['region_count']}")
        print("区域详情:")
        for i, info in enumerate(results['regions_info'], 1):
            print(f"  区域 {i}:")
            print(f"    面积: {info['area']}")
            print(f"    长宽比: {info['aspect_ratio']:.2f}")
            print(f"    中心点: ({info['centroid'][1]:.1f}, {info['centroid'][0]:.1f})")
            print(f"    边界框: {info['bbox']}")

if __name__ == "__main__":
    main() 
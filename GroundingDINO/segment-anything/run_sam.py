import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from skimage.morphology import remove_small_objects

# 配置参数
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/root/autodl-tmp/fc-clip/segment-anything/sam_vit_h_4b8939.pth"
BINARY_MASKS_DIR = "/root/autodl-tmp/fc-clip/val-mid/road_anomaly/binary_masks"
ORIGINAL_IMAGES_DIR = "/root/autodl-tmp/fc-clip/vol/road_anomaly/original"
OUTPUT_DIR = "/root/autodl-tmp/fc-clip/segment-anything/test"

# 新增参数
MAX_POINTS = 1  # 点的上限
LOWER_REGION_RATIO = 0.55  # 只考虑下方70%的区域

def get_base_name(filename):
    """
    从文件名中提取基础名称，处理可能的多重扩展名和前缀
    """
    # 移除 binary_ 前缀
    if filename.startswith('binary_'):
        filename = filename[7:]
    
    # 移除所有已知的扩展名
    base = filename
    while True:
        name, ext = os.path.splitext(base)
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            break
        base = name
    return base

def filter_edge_noise(mask, edge_width=3, min_area=100):
    """
    过滤边缘噪声并保留主要区域
    """
    # 创建一个与输入掩码相同大小的空白掩码
    h, w = mask.shape
    filtered = np.zeros_like(mask)
    
    # 去除边缘区域
    inner_mask = mask[edge_width:h-edge_width, edge_width:w-edge_width]
    
    # 寻找连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inner_mask)
    
    # 保留面积大于阈值的区域
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[edge_width:h-edge_width, edge_width:w-edge_width][labels == i] = 255
    
    return filtered

def show_mask(mask, ax):
    """在给定的轴上显示掩码"""
    color = np.array([30/255, 144/255, 255/255, 0.6])  # 半透明蓝色
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    """在给定的轴上显示点"""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def extract_points_from_debug_image(debug_image_path):
    """
    从调试图像中提取红色点的坐标
    """
    # 读取调试图像
    debug_image = cv2.imread(debug_image_path)
    if debug_image is None:
        print(f"错误：无法读取调试图像 {debug_image_path}")
        return []
    
    # 转换为HSV颜色空间，便于提取红色点
    hsv = cv2.cvtColor(debug_image, cv2.COLOR_BGR2HSV)
    
    # 定义红色范围
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    # 寻找红色区域的轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 提取每个轮廓的中心点
    points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append([cx, cy])
    
    print(f"从调试图像中提取了 {len(points)} 个点")
    return points

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化SAM模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 处理每个二值掩码
    for mask_name in os.listdir(BINARY_MASKS_DIR):
        if not mask_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 读取并处理二值掩码
        mask_path = os.path.join(BINARY_MASKS_DIR, mask_name)
        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if binary_mask is None:
            print(f"错误：无法读取掩码文件 {mask_name}")
            continue
            
        # 只考虑下方70%的区域
        h, w = binary_mask.shape
        start_y = int(h * (1 - LOWER_REGION_RATIO))  # 从30%处开始
        lower_mask = binary_mask[start_y:, :]
        
        # 二值化处理
        _, lower_mask = cv2.threshold(lower_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 过滤边缘噪声
        filtered_mask = filter_edge_noise(lower_mask, edge_width=3, min_area=100)

        # 生成候选点提示
        points = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_mask)
        
        # 按面积排序区域
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)  # 按面积降序排序
        
        # 选择最大的几个区域（不超过MAX_POINTS个）
        for i, area in areas[:MAX_POINTS]:
            cx, cy = centroids[i]
            # 调整y坐标（加上start_y偏移）
            points.append([int(cx), int(cy + start_y)])
            print(f"区域 {i}: 面积={area}, 点=({int(cx)}, {int(cy + start_y)})")

        if not points:
            print(f"警告：未找到有效点提示 {mask_name}")
            continue

        # 读取原始图像
        base_name = get_base_name(mask_name)
        original_path = os.path.join(ORIGINAL_IMAGES_DIR, f"{base_name}.jpg")
        
        if not os.path.exists(original_path):
            # 尝试列出原始图像目录中的文件，查找可能的匹配
            print(f"错误：找不到原始图像 {mask_name}")
            print(f"尝试查找的路径: {original_path}")
            print(f"基础名称: {base_name}")
            
            # 尝试查找包含基础名称的文件
            matching_files = [f for f in os.listdir(ORIGINAL_IMAGES_DIR) if base_name.lower() in f.lower()]
            if matching_files:
                print(f"找到可能的匹配: {matching_files}")
                # 使用第一个匹配的文件
                original_path = os.path.join(ORIGINAL_IMAGES_DIR, matching_files[0])
                print(f"使用匹配文件: {original_path}")
            else:
                print(f"未找到任何匹配文件")
                continue
            
        image = cv2.imread(original_path)
        if image is None:
            print(f"错误：无法读取原始图像 {original_path}")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 保存调试图像（显示点位置）
        debug_image = image.copy()
        for x, y in points:
            cv2.circle(debug_image, (x, y), 5, (0, 0, 255), -1)  # 红色圆点
        cv2.line(debug_image, (0, start_y), (w, start_y), (0, 255, 0), 2)  # 绿色分界线
        debug_path = os.path.join(OUTPUT_DIR, f"debug_{mask_name}")
        cv2.imwrite(debug_path, debug_image)
        
        # 使用SAM进行分割
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.ones(len(points)),
            multimask_output=True
        )

        # 选择最佳掩码
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # 保存分割结果
        output_path = os.path.join(OUTPUT_DIR, f"sam_{mask_name}")
        cv2.imwrite(output_path, best_mask.astype(np.uint8) * 255)
        
        # 可视化结果
        plt.figure(figsize=(15, 10))
        
        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title("原始图像")
        plt.axis('off')
        
        # 带点的调试图像
        debug_img_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, 2)
        plt.imshow(debug_img_rgb)
        plt.title("调试图像（带点）")
        plt.axis('off')
        
        # 分割结果
        plt.subplot(1, 3, 3)
        plt.imshow(image_rgb)
        show_mask(best_mask, plt.gca())
        show_points(np.array(points), np.ones(len(points)), plt.gca())
        plt.title(f"分割结果 (得分: {scores[best_idx]:.3f})")
        plt.axis('off')
        
        # 保存可视化结果
        vis_path = os.path.join(OUTPUT_DIR, f"vis_{mask_name.replace('.jpg', '.png').replace('.jpeg', '.png')}")
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()
        
        print(f"已处理：{mask_name} (点数量: {len(points)}, 最佳掩码得分: {scores[best_idx]:.3f})")

if __name__ == "__main__":
    main()
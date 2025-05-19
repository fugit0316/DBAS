import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import gc

# 配置路径
IMAGE_DIR = "/root/autodl-tmp/GroundingDINO/vol/segment_me/dataset_ObstacleTrack/images"
BBOX_DIR = "/root/autodl-tmp/GroundingDINO/melt/Unexpected_object_on_or_above_the_road_surface/boxes"
OUTPUT_DIR = "/root/autodl-tmp/GroundingDINO/melt/Unexpected_object_on_or_above_the_road_surface"
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualization")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# SAM 模型配置
SAM_CHECKPOINT_PATH = "/root/autodl-tmp/GroundingDINO/segment-anything/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

def convert_xywh_to_xyxy(box, image_width, image_height):
    """将归一化的 [x_center, y_center, width, height] 格式转换为像素坐标的 [x1, y1, x2, y2] 格式"""
    cx, cy, w, h = box
    x1 = int((cx - w / 2) * image_width)
    y1 = int((cy - h / 2) * image_height)
    x2 = int((cx + w / 2) * image_width)
    y2 = int((cy + h / 2) * image_height)
    return [x1, y1, x2, y2]

def clamp_bbox(xyxy_box, image_width, image_height):
    """限制边界框在图像范围内"""
    x1, y1, x2, y2 = xyxy_box
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(1, min(x2, image_width))
    y2 = max(1, min(y2, image_height))
    return [x1, y1, x2, y2]

def process_image(image_file, image_path, base_name, predictor):
    """处理图像"""
    print(f"  处理图像: {image_file}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"  无法读取图像: {image_path}")
        return None
        
    print(f"  图像尺寸: {image.shape}")
    image_height, image_width = image.shape[:2]
    
    # 检查是否存在对应的边界框文件
    bbox_file = os.path.join(BBOX_DIR, f"{base_name}.txt")
    if not os.path.exists(bbox_file):
        print(f"  未找到边界框文件: {bbox_file}，生成全黑掩码")
        # 创建全黑掩码
        return np.zeros((image_height, image_width), dtype=np.uint8)
    
    # 读取边界框信息
    boxes = []
    obstacle_labels = []
    
    print(f"  正在读取边界框文件: {bbox_file}")
    with open(bbox_file, 'r') as f:
        lines = f.readlines()
        print(f"  边界框文件包含 {len(lines)} 行")
        
        for i, line in enumerate(lines):
            parts = line.strip().split()
            print(f"    行 {i+1}: {parts}")
            
            if len(parts) >= 5:  # 确保有足够的部分
                # 提取坐标和标签
                cx, cy, w, h = map(float, parts[:4])
                label = ' '.join(parts[4:])
                
                print(f"    解析为: 坐标=[{cx}, {cy}, {w}, {h}], 标签='{label}'")
                
                # 只处理包含 "object" 或 "obstacle" 的标签
                if "object" in label.lower() or "obstacle" in label.lower():
                    # 转换为像素坐标并限制范围
                    xyxy_box = convert_xywh_to_xyxy([cx, cy, w, h], image_width, image_height)
                    xyxy_box = clamp_bbox(xyxy_box, image_width, image_height)
                    
                    # 检查边界框是否有效
                    if xyxy_box[2] - xyxy_box[0] < 1 or xyxy_box[3] - xyxy_box[1] < 1:
                        print(f"    ✗ 边界框太小 (宽度或高度小于1像素)")
                        continue
                    
                    boxes.append(xyxy_box)
                    obstacle_labels.append(label)
                    print(f"    ✓ 包含 'object' 或 'obstacle'，已添加")
                else:
                    print(f"    ✗ 不包含 'object' 或 'obstacle'，已跳过")
    
    if not boxes:
        print(f"  未找到包含 'object' 或 'obstacle' 的边界框: {image_file}，生成全黑掩码")
        # 创建全黑掩码
        return np.zeros((image_height, image_width), dtype=np.uint8)
        
    print(f"  找到 {len(boxes)} 个边界框")
    
    # 设置 SAM 的输入图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    print(f"  已设置 SAM 输入图像")
    
    # 创建用于可视化的图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title(f"分割结果: {image_file}")
    plt.axis('off')
    
    # 创建合并的掩码
    combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # 处理每个边界框
    for i, (box, label) in enumerate(zip(boxes, obstacle_labels)):
        print(f"  处理边界框 {i+1}: {box}, 标签: {label}")
        
        try:
            # 使用 SAM 生成掩码
            print(f"    调用 SAM 预测...")
            masks, scores, _ = predictor.predict(
                box=np.array(box),
                multimask_output=True
            )
            
            print(f"    SAM 返回了 {len(masks)} 个掩码，得分: {scores}")
            
            # 选择最佳掩码
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            
            print(f"    选择了最佳掩码 (索引 {best_mask_idx})，得分: {scores[best_mask_idx]:.3f}")
            print(f"    掩码中的 True 像素数量: {np.sum(mask)}")
            
            # 更新合并的掩码
            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
            
            # 在可视化图像上显示掩码
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            plt.imshow(mask_image)
            
            # 绘制边界框
            rect = plt.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1], 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none'
            )
            plt.gca().add_patch(rect)
            
            # 添加标签
            plt.text(
                box[0], box[1] - 5, 
                f"{label}: {scores[best_mask_idx]:.2f}", 
                color='white', 
                fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5)
            )
            
        except Exception as e:
            print(f"    处理边界框时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存可视化结果
    vis_path = os.path.join(VIS_DIR, f"{base_name}_segmentation.png")
    plt.savefig(vis_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  可视化结果已保存到: {vis_path}")
    
    # 检查合并的掩码是否为空
    if np.sum(combined_mask) == 0:
        print(f"  警告: 合并的掩码为空 (没有分割区域)")
    
    return combined_mask

def main():
    # 加载 SAM 模型
    print("正在加载 SAM 模型...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    predictor = SamPredictor(sam)
    print(f"SAM 模型已加载到设备: {device}")
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for idx, image_file in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] 处理图像: {image_file}")
        image_path = os.path.join(IMAGE_DIR, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        # 处理图像
        combined_mask = process_image(image_file, image_path, base_name, predictor)
        
        # 如果没有掩码，继续下一张图像
        if combined_mask is None:
            print(f"  无法为 {image_file} 生成掩码，跳过")
            continue
        
        # 保存二值化掩码图像 (255为白色，0为黑色)
        binary_mask = combined_mask * 255
        mask_path = os.path.join(MASK_DIR, f"{base_name}.png")
        cv2.imwrite(mask_path, binary_mask)
        print(f"  二值化掩码已保存到: {mask_path}")
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    print("所有图像处理完成！")

if __name__ == "__main__":
    main()
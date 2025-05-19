import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import gc
import time
from typing import Tuple, List

# 配置路径
IMAGE_DIR = "/root/autodl-tmp/GroundingDINO/vol/segment_me/dataset_ObstacleTrack/images"
BBOX_DIR = "/root/autodl-tmp/GroundingDINO/melt/Unexpected_object_on_or_above_the_road_surface/boxes"
OUTPUT_DIR = "/root/autodl-tmp/GroundingDINO/test/vit_b_OT"
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualization")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# SAM 模型配置
SAM_CHECKPOINT_PATH = "/root/autodl-tmp/GroundingDINO/segment-anything/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数量
    
    @param model: PyTorch模型
    @return: 模型参数总数
    """
    return sum(p.numel() for p in model.parameters())

def estimate_gflops(model: torch.nn.Module, image_shape: Tuple[int, int]) -> float:
    """
    估计模型GFLOPs
    
    @param model: PyTorch模型
    @param image_shape: 输入图像尺寸(高, 宽)
    @return: 估计的GFLOPs
    """
    # 计算参数量
    num_params = count_parameters(model)
    
    # 获取图像尺寸
    h, w = image_shape
    pixels = h * w
    
    # 基准分辨率
    base_pixels = 1024 * 1024  # SAM通常在更高分辨率图像上运行
    
    # 估算GFLOPs: 参考SAM论文中的计算复杂度
    # SAM使用ViT架构，计算复杂度与图像大小和参数量相关
    scale_factor = pixels / base_pixels
    
    # SAM的基础计算量估计 (基于ViT模型特性)
    # 视觉Transformer的计算复杂度与序列长度(像素数)和嵌入维度相关
    # 这里使用参数量和像素数的简化关系进行估计
    base_gflops = num_params * 2 / 1e9  # 每个参数平均参与2次运算
    
    # 对于ViT架构，计算复杂度与输入序列长度（像素数）近似线性关系
    estimated_gflops = base_gflops * scale_factor
    
    return estimated_gflops

def measure_sam_fps(predictor: SamPredictor, num_iterations: int, image_shape: Tuple[int, int]) -> float:
    """
    测量SAM模型的推理FPS
    
    @param predictor: SAM预测器
    @param num_iterations: 测量迭代次数
    @param image_shape: 输入图像尺寸(高, 宽)
    @return: 平均FPS
    """
    print(f"测量SAM模型性能，使用{image_shape[0]}x{image_shape[1]}大小的图像...")
    
    # 创建随机测试图像
    h, w = image_shape
    dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # 预热模型
    print("预热模型...")
    predictor.set_image(dummy_image)
    
    # 创建随机边界框
    # 确保边界框在图像范围内，并且具有合理的大小
    box_width = w // 4
    box_height = h // 4
    box_x = w // 2 - box_width // 2
    box_y = h // 2 - box_height // 2
    test_box = np.array([box_x, box_y, box_x + box_width, box_y + box_height])
    
    # 进行几次预热推理
    for _ in range(3):
        predictor.predict(box=test_box)
    
    # 同步GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 测量时间
    print(f"测量FPS (运行{num_iterations}次)...")
    start_time = time.time()
    
    for _ in range(num_iterations):
        predictor.predict(box=test_box)
        
        # 同步GPU确保准确计时
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_iterations / elapsed_time
    
    return fps

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
    
    # 记录单张图像处理时间（包含设置图像和所有边界框处理）
    img_start_time = time.time()
    
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
            # 记录单个边界框处理时间
            box_start_time = time.time()
            
            # 使用 SAM 生成掩码
            print(f"    调用 SAM 预测...")
            masks, scores, _ = predictor.predict(
                box=np.array(box),
                multimask_output=True
            )
            
            box_end_time = time.time()
            box_process_time = box_end_time - box_start_time
            box_fps = 1.0 / box_process_time if box_process_time > 0 else 0
            print(f"    边界框处理时间: {box_process_time:.4f}秒 (FPS: {box_fps:.2f})")
            
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
    
    # 计算总处理时间和FPS
    img_end_time = time.time()
    img_process_time = img_end_time - img_start_time
    img_fps = 1.0 / img_process_time if img_process_time > 0 else 0
    print(f"  图像总处理时间: {img_process_time:.4f}秒 (FPS: {img_fps:.2f})")
    
    # 保存可视化结果
    vis_path = os.path.join(VIS_DIR, f"{base_name}_segmentation.png")
    plt.savefig(vis_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  可视化结果已保存到: {vis_path}")
    
    # 检查合并的掩码是否为空
    if np.sum(combined_mask) == 0:
        print(f"  警告: 合并的掩码为空 (没有分割区域)")
    
    return combined_mask, img_process_time, img_fps, len(boxes)

def main():
    print("\n========== SAM模型性能测试 ==========")
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"使用GPU: {device_name}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    # 加载 SAM 模型
    print("正在加载 SAM 模型...")
    start_load_time = time.time()
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    predictor = SamPredictor(sam)
    end_load_time = time.time()
    load_time = end_load_time - start_load_time
    print(f"SAM 模型已加载到设备: {device} (加载时间: {load_time:.2f}秒)")
    
    # 计算模型参数量
    params = count_parameters(sam)
    params_m = params / 1e6
    print(f"模型参数量: {params:,} ({params_m:.2f}M)")
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"错误: 在 {IMAGE_DIR} 中未找到图像文件")
        return
        
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 计算平均图像尺寸
    avg_h, avg_w = 0, 0
    sample_size = min(5, len(image_files))
    valid_samples = 0
    
    for i in range(sample_size):
        image_path = os.path.join(IMAGE_DIR, image_files[i])
        try:
            img = cv2.imread(image_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            avg_h += h
            avg_w += w
            valid_samples += 1
        except Exception as e:
            print(f"读取图像 {image_files[i]} 出错: {e}")
    
    if valid_samples > 0:
        avg_h //= valid_samples
        avg_w //= valid_samples
        print(f"平均图像尺寸: {avg_h}x{avg_w}")
        
        # 估计GFLOPs
        gflops = estimate_gflops(sam, (avg_h, avg_w))
        print(f"估计GFLOPs: {gflops:.2f}")
        
        # 测量FPS (使用较小的迭代次数避免OOM)
        fps = measure_sam_fps(predictor, 10, (avg_h, avg_w))
        print(f"理论FPS (空载): {fps:.2f}")
    
    # 跟踪性能数据
    processing_start_time = time.time()
    process_times = []
    fps_values = []
    box_counts = []
    processed_count = 0
    
    # 处理每个图像
    for idx, image_file in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] 处理图像: {image_file}")
        image_path = os.path.join(IMAGE_DIR, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        # 处理图像
        try:
            result = process_image(image_file, image_path, base_name, predictor)
            if result is None:
                print(f"  无法为 {image_file} 生成掩码，跳过")
                continue
                
            combined_mask, process_time, img_fps, box_count = result
            
            # 记录性能数据
            process_times.append(process_time)
            fps_values.append(img_fps)
            box_counts.append(box_count)
            processed_count += 1
            
            # 保存二值化掩码图像 (255为白色，0为黑色)
            binary_mask = combined_mask * 255
            mask_path = os.path.join(MASK_DIR, f"{base_name}.png")
            cv2.imwrite(mask_path, binary_mask)
            print(f"  二值化掩码已保存到: {mask_path}")
            
        except Exception as e:
            print(f"  处理图像 {image_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    # 计算总处理时间
    total_processing_time = time.time() - processing_start_time
    
    # 计算平均性能指标
    if process_times:
        avg_process_time = sum(process_times) / len(process_times)
        avg_fps = sum(fps_values) / len(fps_values)
        avg_box_count = sum(box_counts) / len(box_counts) if box_counts else 0
        
        # 计算每个边界框的平均处理时间
        avg_time_per_box = sum([t/max(c, 1) for t, c in zip(process_times, box_counts)]) / len(process_times) if process_times else 0
    else:
        avg_process_time = 0
        avg_fps = 0
        avg_box_count = 0
        avg_time_per_box = 0
    
    # 输出GPU信息
    if torch.cuda.is_available():
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_mem_used = torch.cuda.memory_allocated(0) / 1024**3  # GB
        print(f"\n计算设备: {device_name}")
        print(f"GPU内存: 已用 {gpu_mem_used:.2f}GB / 总计 {gpu_mem_total:.2f}GB")
    
    # 输出性能汇总
    print("\n========== 性能指标汇总 ==========")
    print(f"模型参数量: {params_m:.2f}M")
    if valid_samples > 0:
        print(f"估计GFLOPs (图像大小 {avg_h}x{avg_w}): {gflops:.2f}")
        print(f"理论FPS (空载): {fps:.2f}")
    print(f"平均单图处理时间: {avg_process_time*1000:.2f}ms")
    print(f"平均每个边界框处理时间: {avg_time_per_box*1000:.2f}ms")
    print(f"平均FPS: {avg_fps:.2f}")
    print(f"平均每张图像边界框数量: {avg_box_count:.2f}")
    print(f"总处理时间: {total_processing_time:.2f}秒")
    print(f"处理图像数量: {processed_count}/{len(image_files)}")
    print("==================================")
    
    print("所有图像处理完成！")

if __name__ == "__main__":
    main()
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import gc

# GroundingDINO 导入
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.inference import annotate

# 配置参数
GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"

# 输入输出路径
IMAGE_DIR = "/root/autodl-tmp/GroundingDINO/vol/road_anomaly/original"
OUTPUT_DIR = "/root/autodl-tmp/GroundingDINO/melt/road_anomaly "
BBOX_DIR = os.path.join(OUTPUT_DIR, "boxes")  # 保存边界框信息的目录
VIS_DIR = os.path.join(OUTPUT_DIR, "visualization")  # 保存可视化结果的目录

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BBOX_DIR, exist_ok=True)  # 创建边界框目录
os.makedirs(VIS_DIR, exist_ok=True)  # 创建可视化目录

# 推理参数
TEXT_PROMPT = "obstacle"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

def main():
    # 加载 GroundingDINO 模型
    print("正在加载 GroundingDINO 模型...")
    grounding_dino_model = load_model(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for idx, image_file in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] 处理图像: {image_file}")
        image_path = os.path.join(IMAGE_DIR, image_file)
        
        try:
            # GroundingDINO 推理
            image_source, image = load_image(image_path)
            boxes, logits, phrases = predict(
                model=grounding_dino_model, 
                image=image, 
                caption=TEXT_PROMPT, 
                box_threshold=BOX_THRESHOLD, 
                text_threshold=TEXT_THRESHOLD
            )
            
            # 如果没有检测到物体，继续下一张图像
            if len(boxes) == 0:
                print(f"  未检测到物体: {image_file}")
                continue
            
            # 保存边界框信息为文本文件，使用与原始图像相同的文件名
            base_name = os.path.splitext(image_file)[0]  # 获取不带扩展名的文件名
            bbox_txt_path = os.path.join(BBOX_DIR, f"{base_name}.txt")
            
            with open(bbox_txt_path, 'w') as f:
                for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                    # 保存为 [x, y, w, h] 格式和名称
                    cx, cy, w, h = box.tolist()
                    f.write(f"{cx} {cy} {w} {h} {phrase}\n")
            
            print(f"  边界框信息已保存到: {bbox_txt_path}")
            
            # 生成可视化结果
            # 方法1: 使用 GroundingDINO 的 annotate 函数
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            
            # 检查 annotated_frame 的类型并相应处理
            if isinstance(annotated_frame, np.ndarray):
                # 如果是 numpy 数组，转换为 PIL 图像
                # 确保颜色通道顺序正确 (RGB)
                if annotated_frame.shape[2] == 3:  # 如果有3个颜色通道
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    annotated_pil = Image.fromarray(annotated_frame_rgb)
                else:
                    annotated_pil = Image.fromarray(annotated_frame)
                
                vis_path = os.path.join(VIS_DIR, f"{base_name}.jpg")
                annotated_pil.save(vis_path)
            else:
                # 如果已经是 PIL 图像，直接保存
                vis_path = os.path.join(VIS_DIR, f"{base_name}.jpg")
                annotated_frame.save(vis_path)
            
            print(f"  可视化结果已保存到: {vis_path}")
            
            # 方法2: 使用 matplotlib 生成更详细的可视化
            # 确保图像是RGB格式
            image_rgb = np.array(image_source)
            if image_rgb.shape[2] == 3:  # 如果有3个颜色通道
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            
            # 绘制每个边界框
            for box, logit, phrase in zip(boxes, logits, phrases):
                # 获取边界框坐标
                cx, cy, w, h = box.tolist()
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                
                # 绘制边界框
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                # 添加标签
                plt.text(
                    x1, y1, 
                    f"{phrase}: {logit:.2f}", 
                    color='white', 
                    fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.5)
                )
            
            plt.title(f"检测结果: {image_file}")
            plt.axis('off')
            
            # 保存可视化图像
            #vis_path_detailed = os.path.join(VIS_DIR, f"{base_name}_detailed.jpg")
            #plt.savefig(vis_path_detailed, bbox_inches='tight', dpi=200)
            # plt.close()
            
            #print(f"  详细可视化结果已保存到: {vis_path_detailed}")
            
            # 清理内存
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f"  已完成: {image_file} (检测到 {len(boxes)} 个物体)")
            
        except Exception as e:
            print(f"  处理 {image_file} 时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            continue
    
    print("所有图像处理完成！")

if __name__ == "__main__":
    main() 
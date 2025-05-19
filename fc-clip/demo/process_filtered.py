import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

"""
 * @description 已知类别列表
 */
"""
KNOWN_CLASSES = {
    "road", "railroad",  # 道路相关
    "sidewalk", "pavement",  # 人行道
    "building", "buildings", "edifice", "edifices", "house", "ceiling",  # 建筑物
    "wall", "walls", "brick wall", "stone wall", "tile wall", "wood wall",  # 墙壁
    "fence", "fences",  # 围栏
    "pole", "poles",  # 杆子
    "traffic light", "traffic lights",  # 交通灯
    "traffic sign", "stop sign",  # 交通标志
    "vegetation", "tree", "trees", "palm tree", "bushes",  # 植被
    "terrain", "river", "sand", "sea", "snow", "water", "mountain", "grass", "dirt", "rock",  # 地形
    "sky", "clouds",  # 天空
    "person",  # 人
    "rider",  # 骑行者
    "car", "cars",  # 汽车
    "truck", "trucks",  # 卡车
    "bus", "buses",  # 公交车
    "train", "trains", "locomotive", "locomotives", "freight train",  # 火车
    "motorcycle", "motorcycles",  # 摩托车
    "bicycle", "bicycles", "bike", "bikes"  # 自行车
}

def create_mask(image):
    """
     * @description 创建图像掩码，检测红色区域
     * @param {np.ndarray} image - 输入图像
     * @return {np.ndarray} - 二值掩码
     */
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义红色的HSV范围
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # 创建红色区域的掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 使用形态学操作改善掩码质量
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def apply_red_mask(image, mask):
    """
     * @description 将红色掩码应用到图像上
     * @param {np.ndarray} image - 原始图像
     * @param {np.ndarray} mask - 二值掩码
     * @return {np.ndarray} - 添加红色掩码后的图像
     */
     """
    # 创建红色遮罩
    red_mask = np.zeros_like(image)
    red_mask[mask > 0] = [0, 0, 255]  # BGR格式
    
    # 将红色遮罩与原图混合
    alpha = 0.5  # 透明度
    result = cv2.addWeighted(image, 1, red_mask, alpha, 0)
    
    return result

def process_images():
    """
     * @description 处理所有过滤后的图像
     */
     """
    # 设置输入输出路径
    input_dir = "/root/autodl-tmp/fc-clip/vol_test/test_cityyaml/road_anomaly/filtered"
    output_dir = "/root/autodl-tmp/fc-clip/vol_test/test_cityyaml/road_anomaly/deal"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每张图片
    for image_file in tqdm(image_files, desc="处理图像"):
        # 读取图像
        input_path = os.path.join(input_dir, image_file)
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"无法读取图像: {input_path}")
            continue
        
        # 创建掩码
        mask = create_mask(image)
        
        # 应用红色掩码
        result = apply_red_mask(image, mask)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"processed_{image_file}")
        cv2.imwrite(output_path, result)
        
        print(f"已处理: {image_file}")

if __name__ == "__main__":
    print("开始处理图像...")
    process_images()
    print("处理完成！") 
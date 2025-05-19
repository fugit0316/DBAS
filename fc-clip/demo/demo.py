"""
This file may have been modified by Bytedance Ltd. and/or its affiliates ("Bytedance's Modifications").
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
"""

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
import cv2
import tqdm
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.engine import launch

from fcclip import add_maskformer2_config, add_fcclip_config
from predictor import VisualizationDemo, OpenVocabVisualizer


# constants
WINDOW_NAME = "fc-clip demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="fcclip demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_coco.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='output',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def count_parameters(model):
    """
    计算模型的参数量
    
    @param model: 要计算参数量的模型
    @return: 模型总参数量
    """
    return sum(p.numel() for p in model.parameters())


def estimate_gflops(model, input_shape):
    """
    估计模型的GFLOPs
    
    @param model: 模型
    @param input_shape: 输入形状 (C, H, W)
    @return: 估计的GFLOPs值
    """
    # 获取参数量
    num_params = count_parameters(model)
    
    # 计算输入图像的像素数
    h, w = input_shape[1], input_shape[2]
    pixels = h * w
    
    # 基准分辨率
    base_pixels = 224 * 224
    
    # 估算GFLOPs: 
    # 1. 每个参数平均参与2次运算
    # 2. 按照图像像素数比例缩放
    scale_factor = pixels / base_pixels
    base_gflops = num_params * 2 / 1e9
    estimated_gflops = base_gflops * scale_factor
    
    return estimated_gflops


def main(args):
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("参数: " + str(args))

    cfg = setup_cfg(args)
    
    # 设置输入和输出路径
    input_path = "/root/autodl-tmp/fc-clip/vol/segment_me/dataset_ObstacleTrack/images"
    output_path = "/root/autodl-tmp/fc-clip/test"
    
    # 创建 VisualizationDemo 实例时传入输出路径
    demo = VisualizationDemo(cfg=cfg, output_path=output_path)  # 使用关键字参数
    
    # 计算模型参数量
    model = demo.predictor.model
    total_params = count_parameters(model)
    params_m = total_params / 1e6
    logger.info(f"模型参数量: {total_params:,} ({params_m:.2f}M)")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "filtered"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 如果没有图片，则退出
    if not image_files:
        logger.info("未找到图像文件")
        return
    
    # 计算平均图像大小，用于估计GFLOPs
    avg_h, avg_w = 0, 0
    for image_file in image_files[:min(10, len(image_files))]:  # 采样最多10张图片
        img_path = os.path.join(input_path, image_file)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        avg_h += h
        avg_w += w
    
    avg_h = avg_h // min(10, len(image_files))
    avg_w = avg_w // min(10, len(image_files))
    logger.info(f"平均图像大小: {avg_h}x{avg_w}")
    
    # 估计GFLOPs
    input_shape = (3, avg_h, avg_w)
    gflops = estimate_gflops(model, input_shape)
    logger.info(f"估计GFLOPs: {gflops:.2f}")
    
    # 测量FPS
    logger.info("开始计算FPS...")
    
    # 预热模型
    warmup_img = read_image(os.path.join(input_path, image_files[0]), format="BGR")
    for _ in range(5):
        _ = demo.run_on_image(warmup_img, "warmup.jpg")
    
    # 记录处理时间
    start_time = time.time()
    num_images = min(50, len(image_files))  # 最多处理50张图片用于FPS计算
    
    # 处理图片并计时
    total_img_time = 0.0
    for image_file in tqdm.tqdm(image_files[:num_images]):
        input_file = os.path.join(input_path, image_file)
        
        # 读取并处理图像
        img = read_image(input_file, format="BGR")
        
        # 记录单张图片的处理时间
        img_start_time = time.time()
        vis_output_original, vis_output_filtered, binary_mask = demo.run_on_image(img, image_file)
        img_end_time = time.time()
        img_process_time = img_end_time - img_start_time
        total_img_time += img_process_time
        
        logger.info(f"图片 {image_file} 的推理时间: {img_process_time:.4f}秒")
    
    # 计算FPS
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_images / total_time
    
    # 输出性能指标
    logger.info("\n========== 性能指标汇总 ==========")
    logger.info(f"模型参数量: {params_m:.2f}M")
    logger.info(f"估计GFLOPs: {gflops:.2f}")
    logger.info(f"平均FPS: {fps:.2f}")
    logger.info(f"处理 {num_images} 张图像总时间: {total_time:.2f}秒")
    logger.info(f"平均单张图片推理时间: {total_img_time/num_images:.4f}秒")
    logger.info("==================================")
    
    # 显示GPU信息
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"计算设备: {device_name}")
    
    # 继续处理其余图片
    if num_images < len(image_files):
        remaining_total_time = 0.0
        remaining_count = 0
        for image_file in tqdm.tqdm(image_files[num_images:]):
            input_file = os.path.join(input_path, image_file)
            img = read_image(input_file, format="BGR")
            
            # 记录单张图片的处理时间
            img_start_time = time.time()
            vis_output_original, vis_output_filtered, binary_mask = demo.run_on_image(img, image_file)
            img_end_time = time.time()
            img_process_time = img_end_time - img_start_time
            remaining_total_time += img_process_time
            remaining_count += 1
            
            logger.info(f"图片 {image_file} 的推理时间: {img_process_time:.4f}秒")
            logger.info(f"已处理并保存: {image_file}")
        
        if remaining_count > 0:
            logger.info(f"剩余 {remaining_count} 张图片的平均推理时间: {remaining_total_time/remaining_count:.4f}秒")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    print("命令行参数:", args)
    launch(
        main,
        1,
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:{}".format(
            torch.randint(2000, 10000, (1,)).item()
        ),
        args=(args,),
    )

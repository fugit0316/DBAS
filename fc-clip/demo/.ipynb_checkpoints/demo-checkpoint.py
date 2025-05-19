"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
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


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def main(args):
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    
    # 设置输入和输出路径
    input_path = "/root/autodl-tmp/fc-clip/vol/road_anomaly/original"
    output_path = "/root/autodl-tmp/fc-clip/test"
    
    # 创建 VisualizationDemo 实例时传入输出路径
    demo = VisualizationDemo(cfg=cfg, output_path=output_path)  # 使用关键字参数

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "filtered"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 处理每张图片
    for image_file in tqdm.tqdm(image_files):
        input_file = os.path.join(input_path, image_file)
        
        # 读取并处理图像
        img = read_image(input_file, format="BGR")
        vis_output_original, vis_output_filtered, binary_mask = demo.run_on_image(img, image_file)
        
        logger.info(f"已处理并保存: {image_file}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    print("Command Line Args:", args)
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

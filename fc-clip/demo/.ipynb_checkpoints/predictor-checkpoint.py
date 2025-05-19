"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/predictor.py
"""

import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch
import itertools
import os
import tqdm
import numpy as np


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor as d2_defaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
import detectron2.utils.visualizer as d2_visualizer


class DefaultPredictor(d2_defaultPredictor):

    def set_metadata(self, metadata):
        self.model.set_metadata(metadata)


class OpenVocabVisualizer(Visualizer):
    def draw_panoptic_seg(self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7):
        """
        Draw panoptic prediction annotations or results.

        Args:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
                segment.
            segments_info (list[dict] or None): Describe each segment in `panoptic_seg`.
                If it is a ``list[dict]``, each dict contains keys "id", "category_id".
                If None, category id of each pixel is computed by
                ``pixel // metadata.label_divisor``.
            area_threshold (int): stuff segments with less than `area_threshold` are not drawn.

        Returns:
            output (VisImage): image object with visualizations.
        """
        pred = d2_visualizer._PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))
        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            text = self.metadata.stuff_classes[category_idx].split(',')[0]
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=d2_visualizer._OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        # draw mask for all instances second
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]

        try:
            scores = [x["score"] for x in sinfo]
        except KeyError:
            scores = None
        stuff_classes = self.metadata.stuff_classes
        stuff_classes = [x.split(',')[0] for x in stuff_classes]
        labels = d2_visualizer._create_text_labels(
            category_ids, scores, stuff_classes, [x.get("iscrowd", 0) for x in sinfo]
        )

        try:
            colors = [
                self._jitter([x / 255 for x in self.metadata.stuff_colors[c]]) for c in category_ids
            ]
        except AttributeError:
            colors = None
        self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

        return self.output


class VisualizationDemo(object):
    def __init__(self, cfg, output_path, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            output_path (str): 输出根目录路径
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        
        # 创建输出目录
        self.output_dirs = {
            'original': os.path.join(output_path, "original"),
            'filtered': os.path.join(output_path, "filtered"),
            'masks': os.path.join(output_path, "masks"),
            'binary_masks': os.path.join(output_path, "binary_masks")  # 新增二值化掩码目录
        }
        
        # 创建所有需要的子目录
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        # 定义已知类别（正常物体）
        self.known_categories = {
            "road", "railroad",  # 道路相关
            "sidewalk", "pavement",  # 人行道
            "building", "buildings", "edifice", "edifices", "house", "ceiling",  # 建筑物
            "window",
            "banner",
            "wall", "walls", "brick wall", "stone wall", "tile wall", "wood wall", "door", "chair", "tent", "wall",  "potted plant", # 墙壁
            "roof",
            "building",
            "fence", "fences", "bench", # 围栏
            #"pole", "poles",  # 杆子
            # "traffic light", "traffic lights",  # 交通灯
            # "traffic sign", "stop sign",  # 交通标志
            "vegetation", "tree", "trees", "palm tree", "bushes",  # 植被
           "terrain", "river", "sand", "sea", "snow", "water", "mountain", "grass", "dirt", "gravel",  # 地形
            "sky", "clouds",  # 天空
            # "person", "handbag",  # 人
           #  "rider",  # 骑行者
            # "car", "cars",  # 汽车
            "rock",
            # "umbrella",
            # "truck", "trucks",  # 卡车
            # "bus", "buses",  # 公交车
            # "train", "trains", "locomotive", "locomotives", "freight train",  # 火车
            # "motorcycle", "motorcycles",  # 摩托车
            # "bicycle", "bicycles", "bike", "bikes", # 自行车
            

        }

        # 设置OOD（异常）的显示颜色为红色
        self.ood_color = [1.0, 0.0, 0.0]  # RGB格式，红色
        
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def filter_segments_info(self, segments_info):
        """过滤 segments_info，只保留未知类别"""
        filtered_info = []
        print("\n=== 开始过滤分割结果 ===")
        
        for segment in segments_info:
            category_id = segment["category_id"]
            category_name = self.metadata.stuff_classes[category_id].split(',')[0].strip().lower()
            
            # 计算区域面积（如果没有提供的话）
            if 'area' not in segment:
                # 使用 panoptic_seg 计算面积
                mask = (self.current_panoptic_seg == segment["id"])
                segment['area'] = int(mask.sum().item())
            
            # 打印每个检测到的类别
            print(f"检测到类别: {category_name} (ID: {category_id}, 面积: {segment['area']})")
            
            # 如果类别不在已知列表中，且面积不是特别大，保留该segment
            if (category_name.lower() not in {k.lower() for k in self.known_categories} and 
                segment['area'] < 50000000000):  # 面积阈值
                filtered_info.append(segment)
                print(f"✓ 保留未知类别: {category_name}")
            else:
                print(f"✗ 过滤掉类别: {category_name}")
        
        print(f"\n过滤前类别数量: {len(segments_info)}")
        print(f"过滤后类别数量: {len(filtered_info)}")
        print("=== 过滤完成 ===\n")
        
        return filtered_info

    def run_on_image(self, image, image_name):
        """
        * @param image: 输入图像
        * @param image_name: 图像文件名（用于保存）
        * @return: 三种可视化结果
        */
        """
        predictions = self.predictor(image)
        
        # Convert image from OpenCV BGR format to RGB format
        image = image[:, :, ::-1]
        
        # 确保输出文件名为 PNG 格式
        base_name = os.path.splitext(image_name)[0]  # 移除原始扩展名
        output_name = f"{base_name}.png"  # 添加 .png 扩展名
        
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            self.current_panoptic_seg = panoptic_seg
            
            # 过滤 segments_info，只保留未知类别
            filtered_segments = self.filter_segments_info(segments_info)
            
            # 1. 生成并保存原始分割效果
            visualizer_original = OpenVocabVisualizer(image, self.metadata, instance_mode=self.instance_mode)
            vis_output_original = visualizer_original.draw_panoptic_seg(
                panoptic_seg.to(self.cpu_device),
                segments_info,
                alpha=0.7
            )
            vis_output_original.save(os.path.join(self.output_dirs['original'], f"orig_{output_name}"))
            
            # 2. 生成并保存过滤后的异常检测效果
            visualizer_filtered = OpenVocabVisualizer(image, self.metadata, instance_mode=self.instance_mode)
            if filtered_segments:
                vis_output_filtered = visualizer_filtered.draw_panoptic_seg(
                    panoptic_seg.to(self.cpu_device),
                    filtered_segments,
                    alpha=0.7
                )
            else:
                vis_output_filtered = visualizer_filtered.output
            vis_output_filtered.save(os.path.join(self.output_dirs['filtered'], f"filt_{output_name}"))
            
            # 3. 生成并保存二值掩码（用于计算IoU和F1）
            binary_mask = torch.zeros_like(panoptic_seg, dtype=torch.uint8)
            if filtered_segments:
                for segment in filtered_segments:
                    mask = (panoptic_seg == segment["id"])
                    binary_mask[mask] = 255

            
            # 将掩码转换为numpy数组并保存为PNG
            binary_mask_np = binary_mask.cpu().numpy().astype(np.uint8)
            cv2.imwrite(
                os.path.join(self.output_dirs['masks'], f"mask_{output_name}"),
                binary_mask_np,
                [cv2.IMWRITE_PNG_COMPRESSION, 9]  # 使用PNG压缩
            )
            
            # 4. 生成并保存二值化掩码（修改逻辑）
            # 初始化为白色（255）
            binary_mask_original = torch.ones_like(panoptic_seg, dtype=torch.uint8) * 255
            
            # 创建已知类别的ID集合
            known_category_ids = set()
            for segment in segments_info:
                category_id = segment["category_id"]
                category_name = self.metadata.stuff_classes[category_id].split(',')[0].strip().lower()
                if category_name.lower() in {k.lower() for k in self.known_categories}:
                    known_category_ids.add(segment["id"])
            
            # 将已知类别区域置为黑色（0），保留未知类别和无掩码区域为白色（255）
            for segment_id in known_category_ids:
                mask = (panoptic_seg == segment_id)
                binary_mask_original[mask] = 0
            
            # 将二值化掩码转换为numpy数组并保存为PNG
            binary_mask_original_np = binary_mask_original.cpu().numpy().astype(np.uint8)
            cv2.imwrite(
                os.path.join(self.output_dirs['binary_masks'], f"binary_{output_name}"),
                binary_mask_original_np,
                [cv2.IMWRITE_PNG_COMPRESSION, 9]  # 使用PNG压缩
            )
            
            print(f"已保存处理结果：{output_name}")
            print(f"- 原始分割：{self.output_dirs['original']}/orig_{output_name}")
            print(f"- 异常分割：{self.output_dirs['filtered']}/filt_{output_name}")
            print(f"- 二值掩码：{self.output_dirs['masks']}/mask_{output_name}")
            print(f"- 二值化掩码：{self.output_dirs['binary_masks']}/binary_{output_name}")
            
            return vis_output_original, vis_output_filtered, binary_mask
        
        return None, None, None

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

def main(args):
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    output_path = "/path/to/output"
    demo = VisualizationDemo(cfg, output_path)

    # 设置输入和输出路径
    input_path = "/root/autodl-tmp/fc-clip/vol/segment_me/dataset_AnomalyTrack/images"
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 处理每张图片
    for image_file in tqdm.tqdm(image_files):
        input_file = os.path.join(input_path, image_file)
        
        # 读取并处理图像
        img = read_image(input_file, format="BGR")
        predictions, vis_output_original, vis_output_filtered = demo.run_on_image(img, image_file)
        
        # 保存结果
        vis_output_original.save(os.path.join(demo.output_dirs['original'], f"orig_{image_file}"))
        vis_output_filtered.save(os.path.join(demo.output_dirs['filtered'], f"filt_{image_file}"))
        
        logger.info(f"已处理并保存: {image_file}")
        logger.info(f"原始分割结果: {os.path.join(demo.output_dirs['original'], f'orig_{image_file}')}")
        logger.info(f"过滤后结果: {os.path.join(demo.output_dirs['filtered'], f'filt_{image_file}')}")

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
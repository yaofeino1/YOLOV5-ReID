import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from PIL import Image
import glob
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
from extracter import FeatureExtractor

class MultiPersonSearcher:
    def __init__(self, 
                 weights='yolov5s.pt',           # YOLOv5权重文件路径
                 reid_config='model/ft_ResNet50/opts.yaml',  # ReID模型配置文件路径
                 query_folder='query_person',            # 查询图片文件夹
                 device='0' if torch.cuda.is_available() else 'cpu',  # 优先使用GPU
                 img_size=640,                           # 输入图像大小
                 conf_thres=0.25,                        # 检测置信度阈值
                 iou_thres=0.45,                         # NMS IOU阈值
                 match_threshold=1.0):                   # 特征匹配阈值
        
        # 初始化设备
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'

        # 加载YOLOv5模型
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        
        if self.half:
            self.model.half()

        # 初始化其他参数
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.match_threshold = match_threshold
        
        # 初始化ReID特征提取器
        self.extractor = FeatureExtractor(reid_config)
        
        # 存储目标人物特征和ID
        self.target_features = {}  # {person_id: features}
        
        # 获取类别名称和颜色
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        # 确保模型支持person类别
        if 'person' not in self.names:
            raise ValueError("Model does not support 'person' class. Please use a model trained on COCO dataset or similar.")
        
        # 加载查询图片
        self.load_query_images(query_folder)
        
        # 预热模型
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))

    def load_query_images(self, query_folder):
        """从文件夹加载查询图片"""
        print(f"Loading query images from {query_folder}...")
        
        if not os.path.exists(query_folder):
            raise FileNotFoundError(f"Query folder {query_folder} not found!")
        
        image_paths = glob.glob(os.path.join(query_folder, "*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No jpg images found in {query_folder}")
            
        image_paths.sort()
        
        for img_path in image_paths:
            person_id = int(os.path.splitext(os.path.basename(img_path))[0])
            try:
                target_img = Image.open(img_path)
                features = self.extractor.extract_feature(target_img)
                self.target_features[person_id] = features
                print(f"Loaded person ID: {person_id} from {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.target_features)} query images")

    def process_frame(self, im0s, img):
        """处理单帧图像"""
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(
            pred, 
            self.conf_thres, 
            self.iou_thres, 
            classes=[0],  # 只检测person类别（COCO数据集中的第0类）
            agnostic=False
        )
        t2 = time_synchronized()

        # Process detections
        matched_persons = set()
        for i, det in enumerate(pred):  # 每张图片的检测结果
            im0 = im0s.copy()
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Process each detection
                for *xyxy, conf, cls in reversed(det):
                    if self.names[int(cls)] == 'person':  # 只处理person类别
                        x1, y1, x2, y2 = map(int, xyxy)
                        if x1 < 0 or y1 < 0 or x2 >= im0.shape[1] or y2 >= im0.shape[0]:
                            continue
                            
                        person_img = im0[y1:y2, x1:x2]
                        if person_img.size == 0:
                            continue
                        
                        # 转换为PIL图像并提取特征
                        person_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                        features = self.extractor.extract_feature(person_img)
                        
                        # 特征匹配
                        best_match_id = None
                        min_dist = float('inf')
                        
                        for person_id, target_feat in self.target_features.items():
                            if person_id in matched_persons:
                                continue
                            
                            dist = np.linalg.norm(features - target_feat)
                            if dist < min_dist and dist < self.match_threshold:
                                min_dist = dist
                                best_match_id = person_id
                        
                        # 绘制结果
                        if best_match_id is not None:
                            matched_persons.add(best_match_id)
                            label = f'ID:{best_match_id:04d} ({min_dist:.2f})'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
            
            print(f'Done. ({t2 - t1:.3f}s)')
            return im0

    def search_video(self, source, view_img=True, save_path=None):
        """在视频中搜索目标人物"""
        if not self.target_features:
            raise ValueError("No query images loaded! Please check the query folder.")
            
        # 创建output文件夹
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 处理保存路径
        if save_path is None:
            # 从源文件名生成输出文件名
            if source.isnumeric():
                save_path = os.path.join(output_dir, f'camera_{source}_out.mp4')
            else:
                source_name = os.path.splitext(os.path.basename(source))[0]
                save_path = os.path.join(output_dir, f'{source_name}_out.mp4')
            
        # 设置数据加载器
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
            
        if webcam:
            cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=self.img_size, stride=self.stride)
        else:
            dataset = LoadImages(source, img_size=self.img_size, stride=self.stride)

        vid_path, vid_writer = None, None
        t0 = time.time()
        
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 处理图像
            if webcam:
                for i, im0 in enumerate(im0s):
                    processed_frame = self.process_frame(im0, img)
                    if view_img:
                        cv2.imshow(f'Person Search', processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            return
            else:
                processed_frame = self.process_frame(im0s, img)
                if view_img:
                    cv2.imshow('Person Search', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # 保存结果
            if save_path:
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, processed_frame.shape[1], processed_frame.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(processed_frame)

        if save_path:
            print(f"Results saved to {save_path}")
        print(f'Done. ({time.time() - t0:.3f}s)')
        
        if vid_writer:
            vid_writer.release()
        cv2.destroyAllWindows()

def main():
    # 设置默认参数
    class DefaultArgs:
        def __init__(self):
            # 模型和数据相关
            self.weights = 'yolov5s.pt'  # YOLOv5权重文件路径
            self.query_folder = 'query_person'   # 待查询的人物图片文件夹
            self.source = '/home/jtse/code/tensorrt-yolov5/yolov5_reid/videos/Double1.mp4'            # 默认视频源
            
            # 检测参数
            self.img_size = 640                 # 输入图像大小
            self.conf_thres = 0.25              # 检测置信度阈值
            self.iou_thres = 0.45               # NMS IOU阈值
            self.match_threshold = 1.0          # 特征匹配阈值
            
            # 设备和显示相关
            self.device = '0' if torch.cuda.is_available() else 'cpu'  # 优先使用GPU
            self.view_img = True                # 是否显示结果
            self.save_path = None               # 结果保存路径将自动生成
    
    # 使用默认参数
    opt = DefaultArgs()
    
    # 如果需要从命令行修改参数，可以取消注释下面的代码
    """
    parser = argparse.ArgumentParser(description='Multi-Person Search in Video')
    parser.add_argument('--weights', type=str, default=opt.weights, help='model.pt path')
    parser.add_argument('--query-folder', type=str, default=opt.query_folder, help='query images folder')
    parser.add_argument('--source', type=str, default=opt.source, help='source')
    parser.add_argument('--img-size', type=int, default=opt.img_size, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=opt.conf_thres, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=opt.iou_thres, help='NMS IoU threshold')
    parser.add_argument('--device', default=opt.device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default=opt.view_img, help='display results')
    parser.add_argument('--save-path', type=str, default=opt.save_path, help='save results to *.mp4')
    parser.add_argument('--match-threshold', type=float, default=opt.match_threshold, help='feature matching threshold')
    opt = parser.parse_args()
    """

    with torch.no_grad():
        searcher = MultiPersonSearcher(
            weights=opt.weights,
            device=opt.device,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            match_threshold=opt.match_threshold,
            query_folder=opt.query_folder
        )
        
        searcher.search_video(opt.source, opt.view_img, opt.save_path)

if __name__ == '__main__':
    main()
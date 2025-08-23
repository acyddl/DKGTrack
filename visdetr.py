# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os

os.environ['CUDA_VISIBLE_DEVICES']='2'

import numpy as np
import random
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from models import build_model
from util.tool import load_model
from main import get_args_parser
from torch.nn.functional import interpolate
from typing import List
from util.evaluation import Evaluator
from thop import profile
from thop import clever_format
import motmetrics as mm
import torch.nn as nn
import shutil
import json
import matplotlib.pyplot as plt

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util import box_ops, checkpoint
from einops import rearrange, repeat
torch.set_grad_enabled(False)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]


def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img.numpy(), c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img.numpy(), c1, c2, color, -1)  # filled
        cv2.putText(img.numpy(),
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img.numpy(), score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)
    return img


'''
deep sort 中的画图方法，在原图上进行作画
'''


def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        if identities[i] == 118 :
            continue
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[1] == 2, 'invalid points shape: {}'.format(points.shape)
    for i, (x, y) in enumerate(points):
        if i >= 300:
            color = (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 2, color=color, thickness=2)
    return img


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def load_label(label_path: str, img_size: tuple) -> dict:
    labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)
    h, w = img_size
    # Normalized cewh to pixel xyxy format
    labels = labels0.copy()
    labels[:, 2] = w * (labels0[:, 2])
    labels[:, 3] = h * (labels0[:, 3])
    labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4])
    labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5])
    targets = {'boxes': [], 'labels': [], 'area': []}
    num_boxes = len(labels)

    visited_ids = set()
    for label in labels[:num_boxes]:
        obj_id = label[1]
        if obj_id in visited_ids:
            continue
        visited_ids.add(obj_id)
        targets['boxes'].append(label[2:6].tolist())
        targets['area'].append(label[4] * label[5])
        targets['labels'].append(0)
    targets['boxes'] = np.asarray(targets['boxes'])
    targets['area'] = np.asarray(targets['area'])
    targets['labels'] = np.asarray(targets['labels'])
    return targets


def filter_pub_det(res_file, pub_det_file, filter_iou=False):
    frame_boxes = {}
    with open(pub_det_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                continue
            elements = line.strip().split(',')
            frame_id = int(elements[0])
            x1, y1, w, h = elements[2:6]
            x1, y1, w, h = float(x1), float(y1), float(w), float(h)
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            if frame_id not in frame_boxes:
                frame_boxes[frame_id] = []
            frame_boxes[frame_id].append([x1, y1, x2, y2])

    for frame, boxes in frame_boxes.items():
        frame_boxes[frame] = np.array(boxes)

    ids = {}
    num_filter_box = 0
    with open(res_file, 'r') as f:
        lines = f.readlines()
    with open(res_file, 'w') as f:
        for line in lines:
            if len(line) == 0:
                continue
            elements = line.strip().split(',')
            frame_id, obj_id = elements[:2]
            frame_id = int(frame_id)
            obj_id = int(obj_id)
            x1, y1, w, h = elements[2:6]
            x1, y1, w, h = float(x1), float(y1), float(w), float(h)
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            if obj_id not in ids:
                # track initialization.
                if frame_id not in frame_boxes:
                    num_filter_box += 1
                    print("filter init box {} {}".format(frame_id, obj_id))
                    continue
                pub_dt_boxes = frame_boxes[frame_id]
                dt_box = np.array([[x1, y1, x2, y2]])
                if filter_iou:
                    max_iou = bbox_iou(dt_box, pub_dt_boxes).max()
                    if max_iou < 0.5:
                        num_filter_box += 1
                        print("filter init box {} {}".format(frame_id, obj_id))
                        continue
                else:
                    pub_dt_centers = (pub_dt_boxes[:, :2] + pub_dt_boxes[:, 2:4]) * 0.5
                    x_inside = (dt_box[0, 0] <= pub_dt_centers[:, 0]) & (dt_box[0, 2] >= pub_dt_centers[:, 0])
                    y_inside = (dt_box[0, 1] <= pub_dt_centers[:, 1]) & (dt_box[0, 3] >= pub_dt_centers[:, 1])
                    center_inside: np.ndarray = x_inside & y_inside
                    if not center_inside.any():
                        num_filter_box += 1
                        print("filter init box {} {}".format(frame_id, obj_id))
                        continue
                print("save init track {} {}".format(frame_id, obj_id))
                ids[obj_id] = True
            f.write(line)

    print("totally {} boxes are filtered.".format(num_filter_box))


class ListImgDataset(Dataset):
    def __init__(self, img_list) -> None:
        super().__init__()
        self.img_list = img_list

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        label_path = f_path.replace('training', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
        # print(label_path)
        cur_img = cv2.imread(f_path)
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        targets = load_label(label_path, cur_img.shape[:2]) if os.path.exists(label_path) else None
        # img = draw_bboxes(torch.tensor(cur_img), targets['boxes'])
        return cur_img, targets

    def init_img(self, img):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, targets = self.load_img_from_file(self.img_list[index])
        return self.init_img(img)


# 在列维度上进行归一化
def normalize_columns(tensor):
    # 计算每列的最小值和最大值
    col_min = tensor.min(dim=1, keepdim=True)[0]
    col_max = tensor.max(dim=1, keepdim=True)[0]

    # 归一化
    normalized_tensor = (tensor - col_min) / (col_max - col_min + 1e-8)  # 加小值防止除以零
    return normalized_tensor


def get_attention_map(model, ori_img, input_image, target_size, sentence, track_instances):
    # if track_instances is None:
    #     track_instances = model._generate_empty_tracks()

    # if not isinstance(input_image, NestedTensor):
    #     img = nested_tensor_from_tensor_list(input_image.cuda().float())

    # with torch.no_grad():
    #     out = model._forward_single_image(img,track_instances,sentence)
    #     attention_map = out['attn']
    #     box = out['pred_boxes']
    # # 假设注意力图是一个2D张量
    # # 这里将其转换为NumPy数组并进行归一化
    # # attention_map = normalize_columns(attention_map.squeeze())
    # attention_map = attention_map.squeeze().cpu().numpy()
    # # attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    map = attention_map[:200, :]

    # 创建子图
    num_subplots = 7
    rows = 7  # 选择行数
    cols = 1  # 选择列数

    # 创建一个新的图形
    plt.figure(figsize=(15, 6))

    # 拆分为 10 个子图，每个子图显示 [10, 7] 的数据
    for i in range(num_subplots):
        # 计算每个子图的数据范围
        # start_idx = i * 10
        # end_idx = start_idx + 20
        # sub_map = map[start_idx:end_idx, :]
        sub_map = map[:, i].reshape(1, -1)
        plt.subplot(rows, cols, i + 1)

        # 绘制子图
        # plt.subplot(rows, cols, i + 1)
        plt.imshow(sub_map, aspect='auto', cmap='jet', alpha=0.5)  # 使用热图色彩
        plt.title(f'Subplot {i + 1}')
        # plt.colorbar()  # 添加颜色条
        plt.axis('off')

    # 调整布局
    plt.tight_layout()
    plt.savefig('figs/savefig_example.png')  # 保存图像
    plt.show()

    threshold = 0.95
    indices = np.where(attention_map[:200, 2] > threshold)[0]

    # convert to [x0, y0, x1, y1] format
    boxes = box_ops.box_cxcywh_to_xyxy(box)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_size
    scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
    boxes = boxes * scale_fct[None, :]
    boxes = boxes.squeeze().cpu().numpy()[indices,:]

    img_show = draw_bboxes(ori_img, boxes, indices)
    cv2.imwrite('figs/185_2.jpg',cv2.cvtColor(img_show.numpy(), cv2.COLOR_BGR2RGB))

    # # 可视化注意力图
    # plt.imshow(attention_map, cmap='jet', alpha=0.5)  # 使用热图色彩
    # plt.axis('off')
    # plt.title('Attention Map')
    # plt.savefig('figs/savefig_example.png')
    # plt.show()
    # return attention_map

def pt_vis(cur_img, sentence,img_h, img_w):
    # sample_location
    # sampling_grids = 2 * sample_location - 1
    # sampling_grid_l_ = sampling_grids[:, :, :, -1].transpose(1, 2).flatten(0, 1)
    res = model.inference_single_image(cur_img.cuda().float(), sentence, (img_h, img_w))
    indices = res['track_instances'].scores[:300] > 0.0
    pt = res['ref_pts'][:,indices.squeeze(),:]
    pt = pt.squeeze().detach().cpu().numpy()
    img = Image.open('/data1/zsp/refer-kitti/KITTI/training/image_02/0004/000001.png')  # 替换为你的原图像路径
    img = np.array(img.convert('RGB'))
    num_points = pt.shape[0]
    colors = np.random.rand(num_points, 3) 
    plt.imshow(img)  # 显示图像
    plt.scatter(pt[:, 0], pt[:, 1], color=colors, marker='o', s=5)  # 绘制点
    plt.axis('off')  # 关闭坐标轴
    plt.title('Points on Image')
    plt.savefig('figs/cpt.jpg')
    plt.show()

def enc_vis(flatten):
    conv_1 = rearrange(flatten[:,:11136,:], 'b (h w) c ->b c h w', h=58, w=192)
    conv_2 = rearrange(flatten[:,11136:13920,:], 'b (h w) c ->b c h w', h=29, w=96)
    conv_3 = rearrange(flatten[:,13920:14640,:], 'b (h w) c ->b c h w', h=15, w=48)
    conv_2 = interpolate(conv_2, size=(58, 192), mode='bilinear', align_corners=False)
    conv_3 = interpolate(conv_3, size=(58, 192), mode='bilinear', align_corners=False)
    conv_features = (conv_1+conv_2+conv_3)[0,:,:,:]/3
    # conv_features = flatten[0,:idx[1],:].view(-1, 58, 192)
    # avg_features = conv_features.mean(dim=0)
    # conv_vis(avg_features)
    conv_vis(conv_features[1,:,:])

def conv_vis(conv_features):
    # resnet50输出可视化
    # average_map = conv_features['0'].tensors.mean(dim=1).detach().cpu().numpy()[0]
    # average_map = conv_features['0'].tensors[0, 200, :, :].squeeze(0).detach().cpu().numpy()
    average_map = conv_features.detach().cpu().numpy()
    # 归一化到0-1之间
    average_map = (average_map - np.min(average_map)) / (np.max(average_map) - np.min(average_map))
    average_map = np.power(average_map, 2)  # 低值变得更低，高值影响不大

    # min_threshold = np.percentile(average_map, 30)  # 仅保留前 70% 的高值部分
    # average_map[average_map < min_threshold] = 0
    # 读取原图像
    original_image = Image.open('/media/data4/lgy/RMOT_project/TempRMOT/data/Dataset/refer-kitti/KITTI/training/image_02/0001/000128.png')  # 替换为你的原图像路径
    # 将热力图转换为带颜色的热力图
    heatmap = plt.cm.jet(average_map)  # 使用 jet colormap
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    # 将热力图转换为PIL图像并调整大小
    heatmap_image = Image.fromarray(heatmap).resize(original_image.size, Image.BILINEAR)
    heatmap_image = heatmap_image.convert('RGB')  # 转换为RGB模式
    # 创建带有透明度的热力图
    heatmap_image = Image.blend(original_image.convert('RGB'), heatmap_image, alpha=0.5)
    # 保存叠加结果
    heatmap_image.save('/media/data4/lgy/RMOT_project/Work_2/TRMOT_work2_3/figs/heatmap_01_128.jpg')

    # 显示叠加结果
    plt.imshow(heatmap_image)
    plt.axis('off')  # 关闭坐标轴
    plt.title('Combined Heatmap with Original Image')
    plt.show()

class WrappedModel(nn.Module):
    def __init__(self, model, track_instances, sentence):
        super().__init__()
        self.model = model
        self.track_instances = track_instances
        self.sentence = sentence

    def forward(self, x):
        # 注意要变成 NestedTensor，因为原始模型的 forward 依赖它
        nt = nested_tensor_from_tensor_list(x)
        return self.model._forward_single_image(nt, self.track_instances, self.sentence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    checkpoint_id = int(args.resume.split('/')[-1].split('.')[0].split('t')[-1])
    model = load_model(detr, args.resume)
    model.eval()
    model = detr.cuda()

    # 创建 dummy 输入
    dummy_input = torch.randn(1, 3, 384, 1280).cuda()  # 替换成你的实际图像尺寸
    dummy_sentence = ['cars-which-are-parking']
    dummy_tracks = detr._generate_empty_tracks()

    # 初始化 WrappedModel
    wrapped_model = WrappedModel(model, dummy_tracks, dummy_sentence)
    wrapped_model = wrapped_model.cuda()

    # 执行 FLOPs 和 Params 测试
    flops, params = profile(wrapped_model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], "%.2f")

    print(f"Params: {params}, FLOPs: {flops}")

    sentence = ['cars-which-are-parking']
    img_list = ['/media/data4/lgy/RMOT_project/TempRMOT/data/Dataset/refer-kitti/KITTI/training/image_02/0001/000128.png']
    loader = DataLoader(ListImgDataset(img_list), 1, num_workers=2)
    for i, (cur_img, ori_img) in enumerate(tqdm(loader)):
        cur_img, ori_img = cur_img[0], ori_img[0]
        img_h, img_w, _ = ori_img.shape

    track_instances = detr._generate_empty_tracks()

    if not isinstance(cur_img, NestedTensor):
        img = nested_tensor_from_tensor_list(cur_img.cuda().float())

    out = model._forward_single_image(img,track_instances,sentence)

    # keep only predictions with 0.7+ confidence
    probas = out['pred_logits'][0, :].sigmoid()
    keep = probas.max(dim=-1).values > 0.007
    # keep=[1,2,3,4,5]
    box = out['pred_boxes'][0, keep]
    # convert to [x0, y0, x1, y1] format
    boxes = box_ops.box_cxcywh_to_xyxy(box)
    # and from relative [0, 1] to absolute [0, height] coordinates
    scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
    bboxes_scaled = boxes * scale_fct[None, :]

    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    input_level_start_index = []
    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[0])
        ),
        # model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        #     lambda self, input, output: input_level_start_index.append(output[1])
        # ),
        # model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
        #     lambda self, input, output: dec_attn_weights.append(output[1])
        # ),
        ]

    # propagate through the model
    out = model._forward_single_image(img,track_instances,sentence)

    # 用完的hook后删除
    for hook in hooks:
        hook.remove()
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0].unsqueeze(0)
    sampling_locations = dec_attn_weights[0]
    # input_level_start_index = input_level_start_index[0]
    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]
    # conv_vis(conv_features['0'].tensors[0, 200, :, :].squeeze(0))
    enc_vis(enc_attn_weights)
    pt_vis(sampling_locations)




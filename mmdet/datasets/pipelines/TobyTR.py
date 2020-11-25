from ..builder import PIPELINES
from copy import deepcopy
from .compose import Compose
import os
import cv2
import numpy as np
import torch

# {'img_info': 
#     {'filename': '/home/../../data3/giangData/image_crop_1175x7680/631.png', 
#     'width': 7680, 
#     'height': 1175, 
#     'ann': 
#         {'bboxes': array([[5420.,  975., 6185., 1724.]], dtype=float32), 
#         'labels': array([0])}}, 
# 'ann_info': 
#     {'bboxes': array([[5420.,  975., 6185., 1724.]], dtype=float32), 
#     'labels': array([0])}, 
# 'img_prefix': 'data/coco/train2017/', 
# 'seg_prefix': None, 'proposal_file': None, 'bbox_fields': [], 
# 'mask_fields': [], 'seg_fields': []}

# {'img_metas': 
#     DataContainer(
#         {'filename': '/home/../../data3/giangData/image_crop_1175x7680/9596.png', 
#         'ori_filename': '/home/../../data3/giangData/image_crop_1175x7680/9596.png',
#          'ori_shape': (1175, 7680, 3), 
#          'img_shape': (204, 1333, 3), 
#          'pad_shape': (224, 1344, 3), 
#          'scale_factor': array([0.17356771, 0.17361702, 0.17356771, 0.17361702], dtype=float32), 
#          'flip': True, 
#          'flip_direction': 'horizontal', 
#          'img_norm_cfg': 
#             {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 
#             'std': array([58.395, 57.12 , 57.375], dtype=float32), 
#             'to_rgb': True}}), 
# 'img': DataContainer(tensor([[[-2.1179, -2.1179, -2.1179,  ...,  0.0000,  0.0000,  0.0000],
#          [-2.1179, -2.1179, -2.1179,  ...,  0.0000,  0.0000,  0.0000],
#          [-2.1179, -2.1179, -2.1179,  ...,  0.0000,  0.0000,  0.0000],
#          ...,
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

#         [[-2.0357, -2.0357, -2.0357,  ...,  0.0000,  0.0000,  0.0000],
#          [-2.0357, -2.0357, -2.0357,  ...,  0.0000,  0.0000,  0.0000],
#          [-2.0357, -2.0357, -2.0357,  ...,  0.0000,  0.0000,  0.0000],
#          ...,
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],

#         [[-1.8044, -1.8044, -1.8044,  ...,  0.0000,  0.0000,  0.0000],
#          [-1.8044, -1.8044, -1.8044,  ...,  0.0000,  0.0000,  0.0000],
#          [-1.8044, -1.8044, -1.8044,  ...,  0.0000,  0.0000,  0.0000],
#          ...,
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]])), 
# 'gt_bboxes': DataContainer(tensor([[568.0871, 111.6357, 937.7863, 204.0000]])), 
# 'gt_labels': DataContainer(tensor([0]))}


@PIPELINES.register_module()
class TobyRead(Compose):
    def __init__(self, *args, **kwags):
        super(TobyRead, self).__init__(*args, **kwags)
        self.add_path = '/home/../../data3/giangData/image_vol1_Sejin/'
    def __call__(self, data):
        # add_path = '/home/../../data3/giangData/image_vol1_Sejin/'
        fake = deepcopy(data)
        # print(fake)
        infor = fake['img_info']['filename']
        name = infor.split('/')[-1]
        add_name = self.add_path + name
        fake['img_info']['filename'] = add_name
        # print(fake)
        for t in self.transforms:
            data = t(data)
            fake = t(fake)
            if data is None:
                return None
        add_tensor = fake['img']._data
        data['img']._data = torch.cat((data['img']._data, add_tensor), dim=0)
        return data


    

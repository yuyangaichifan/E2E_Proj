### yu yang #######
import math
import os
import joblib
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.parse_config import *
from lib.core.config import VIBE_DATA_DIR
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
import numpy as np
from  tqdm import tqdm
import cv2
import time
from lib.utils.yolo_utils import (
    build_targets, to_cpu, non_max_suppression,
    weights_init_normal, load_classes, select_iou_GT)

from lib.models.tracker import MPT
from lib.models.yolo import Darknet

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)
from lib.utils.renderer import Renderer
import colorsys
import shutil
from lib.models.spin import Regressor, hmr


class Det_VIBE(nn.Module):
    def __init__(self, cfg):
        super(Det_VIBE, self).__init__()
        self.detector = Darknet(cfg.YOLO.MODEL_DEF)
        self.detector.apply(weights_init_normal)
        self.detector.load_darknet_weights(cfg.YOLO.PRETRAINED_MODEL)
        self.device = cfg.DEVICE
        self.generator = e2e_VIBE(
            seqlen=cfg.DATASET.SEQLEN,
            n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
            hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
            add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
            use_residual=cfg.MODEL.TGRU.RESIDUAL,
        ).to(cfg.DEVICE)
        self.tracker = MPT(
            device=cfg.DEVICE,
            batch_size=cfg.TRACKER.TRACKER_BATCH_SIZE,
            output_format='dict',
        )
        self.nms = non_max_suppression
        self.class_names = load_classes(cfg.YOLO.CLASS_PATH)


    def forward(self, img_batch, gt_bbox_batch=None):
        tmp_shape = img_batch.shape
        img_batch_yolo = img_batch.view([-1, tmp_shape[2], tmp_shape[3],tmp_shape[4]])
        tmp_shape = gt_bbox_batch.shape
        gt_bbox_yolo = gt_bbox_batch.view([-1, tmp_shape[2]])
        self.detector.cuda()
        if gt_bbox_batch is None:
            self.detector.eval()
            with torch.no_grad():

                yolo_output = self.detector(img_batch_yolo)

                from lib.utils.vis import batch_vis_yolo_raw, batch_vis_yolo_res
                # batch_vis_yolo_raw(img_batch_yolo, yolo_output, 0)
                ## apply nms
                yolo_output = self.nms(yolo_output, 0.3, 0.4)

                batch_vis_yolo_res(img_batch_yolo, yolo_output, self.class_names)
                ## apply sort tracker
                tracking_results = self.tracker.run_tracker_pred(yolo_output)
                for person_id in list(tracking_results.keys()):
                    if tracking_results[person_id]['frames'].shape[0] < 25:
                        del tracking_results[person_id]
                ## gen all bbox input
                for person_id in tqdm(list(tracking_results.keys())):
                    joints2d = None
                    bboxes = tracking_results[person_id]['bbox']
                    frames = tracking_results[person_id]['frames']
        else:
            self.detector.eval()
            yolo_output = self.detector(img_batch_yolo)
            select_iou_GT(yolo_output, gt_bbox_yolo, 0.5)
            print('111')

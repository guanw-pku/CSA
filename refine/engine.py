# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
from json.encoder import py_encode_basestring
import math
import os
import sys
from typing import Iterable
import time
import torch

import pdb
import util.misc as utils

from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from cams_deit import resize_cam, get_bboxes, blend_cam, tensor2image, AveragePrecisionMeter, bgrtensor2image, draw_gt_bbox, get_multi_bboxes
import copy
import torchvision
from util.misc import all_gather, get_rank
from engine_loc import visualize_boxes

from util import box_ops
import numpy as np
from torchvision.ops import RoIAlign
import json
from PIL import Image, ImageDraw  
from models.misc import nms, soft_nms, diou_nms
from ensemble_boxes import weighted_boxes_fusion

import cv2
import matplotlib.pyplot as plt


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    gettime = lambda :time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    header = f'[{gettime()}] => Epoch: [{epoch}]'
    print_freq = 100
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        pseudo_label = get_pseudo_label(outputs, samples, targets, args)
        for t, p in zip(targets, pseudo_label):
            t.update(p)
        loss_dict = criterion(outputs, targets)
        weight_dict = copy.deepcopy(criterion.weight_dict)

        if epoch < 4:
            for k in weight_dict:
                if not 'img_label' in k:
                    weight_dict[k] = 0.0

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        for key in loss_dict:
            if key in weight_dict:
                print(f'loss: {loss_dict[key]}, weight: {weight_dict[key]}')
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_refine(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    args=None, postprocessors=None, criterion_refine=None):
    
    model.train()
    criterion.train()
    criterion_refine.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    gettime = lambda :time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    header = f'[{gettime()}] => Epoch: [{epoch}]'
    print_freq = 100
    weight_dict = copy.deepcopy(criterion.weight_dict)
    rf_header = 'ref'
    weight_dict = get_refine_weight_dict(weight_dict, args.num_refines, header=rf_header)
    # if epoch % 20 == 0:
    #     args.bbox_loss_coef /= 10
    #     args.giou_loss_coef /= 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        pseudo_label = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args)
        for t, p in zip(targets, pseudo_label):
            t.update(p)

        pseudo_label_refine = get_refinements_pseudo_label(outputs, 
                                                    samples, targets, args, postprocessors)
        # print(f'outputs[0].keys(): {outputs[0].keys()}')
        # pdb.set_trace()
        loss_dict = criterion(outputs[0], targets)
        for rf, out in outputs.items():
            if rf == 0:
                continue
            loss_dict_rf = criterion_refine(out, pseudo_label_refine[rf])
            for k, v in loss_dict_rf.items():
                key = f'{rf_header}_{rf}_{k}'
                loss_dict[key] = v
        if epoch < 7:
            for k in weight_dict:
                if not ('img_label' in k or 'drloc' in k):
                    weight_dict[k] = 0.0

        if epoch < 15:
            for k in weight_dict:
                if rf_header in k:
                    weight_dict[k] = 0.0
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        pdb.set_trace()
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_refine_match(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    args=None, postprocessors=None, criterion_refine=None):
    
    model.train()
    criterion.train()
    criterion_refine.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    gettime = lambda :time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    header = f'[{gettime()}] => Epoch: [{epoch}]'
    print_freq = 100
    weight_dict = copy.deepcopy(criterion.weight_dict)
    rf_header = 'ref'
    weight_dict = get_refine_weight_dict(weight_dict, args.num_refines, header=rf_header)
    # if epoch % 20 == 0:
    #     args.bbox_loss_coef /= 10
    #     args.giou_loss_coef /= 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        pseudo_label = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args)

        for t, p in zip(targets, pseudo_label):
            t.update(p)
        
        pseudo_label_refine = get_refinements_pseudo_label(outputs, 
                                                    samples, targets, args, postprocessors)
        loss_dict, match_indices = criterion(outputs[0], targets)
        for rf, out in outputs.items():
            if rf == 0:
                continue
            loss_dict_rf, match_indices = criterion_refine(out, pseudo_label_refine[rf])
            for k, v in loss_dict_rf.items():
                key = f'{rf_header}_{rf}_{k}'
                loss_dict[key] = v
        if epoch < 4:
            for k in weight_dict:
                if not 'img_label' in k:
                    weight_dict[k] = 0.0

        if epoch < 15:
            for k in weight_dict:
                if header in k:
                    weight_dict[k] = 0.0
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        pdb.set_trace()
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_refine_weight_dict(weight_dict, num_refinements, header='rf'):
    tmp_weight_dict = copy.deepcopy(weight_dict)
    org_keys = tmp_weight_dict.keys()
    for rf in range(1, num_refinements+1):
        for key in org_keys:
            new_key = f'{header}_{rf}_{key}'
            weight_dict[new_key] = weight_dict[key]

    return weight_dict


@torch.no_grad()
def get_refinements_pseudo_label(outputs, samples, targets, args, postprocessors):
    targets_refine = {}
    # targets_refine[1] = output_to_pseudo_label(outputs[0], samples, targets, args, postprocessors)
    for k, v in outputs.items():
        if k == args.num_refines:
            break
        targets_refine[k+1] = output_to_pseudo_label(v, samples, targets, args, postprocessors)
    
    return targets_refine

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
        (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def normalize_bbox(boxes, image_size):
    h, w = image_size
    boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.get_device())
    return boxes


@torch.no_grad()
def output_to_pseudo_label(outputs, samples, targets, args, postprocessors):
    # device = samples.tensors.get_device()
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    pdb.set_trace()
    pred_results = postprocessors['bbox'](outputs, orig_target_sizes, targets)

    Pseudo_labels= []
    for idx, result in enumerate(pred_results):
        Pseudo_labels.append(copy.deepcopy(targets[idx]))
        det_cls = result['labels'].detach().clone()
        det_box = result['boxes'].detach().clone()
        det_score=result['scores'].detach().clone()
        Pseudo_labels[-1].update({f'labels':det_cls, 
                        f'boxes': det_box, 
                        f'scores': det_score})
    return Pseudo_labels


@torch.no_grad()
def get_pseudo_label(outputs, samples, targets, args):
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def normalize_bbox(boxes, image_size):
        h, w = image_size
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.get_device())
        return boxes
    device = samples.tensors.get_device()
    cams = outputs['cams_cls']
    cls_logits = outputs['x_logits']
    Pseudo_labels = []
    for batch_i in range(cams.shape[0]):
        image_size_i = samples.tensors.shape[-2:]
        image_label_i = targets[batch_i]['img_label'].data.cpu().numpy().reshape(-1)
        image_score_i = cls_logits[batch_i].sigmoid().data.cpu().numpy().reshape(-1)
        image_score_i = cls_logits[batch_i].reshape(-1)
        estimated_bbox = []
        estimated_class= []
        for class_i in range(args.num_classes):
            if image_label_i[class_i] > 0:
                cam_i = cams[batch_i, [class_i], :, :]
                cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                cam_i = resize_cam(cam_i, size=image_size_i)
                bbox = get_bboxes(cam_i, cam_thr=args.cam_thr)
                bbox = torch.tensor(bbox)
                bbox = box_xyxy_to_cxcywh(bbox)
                estimated_bbox.append(bbox)
                estimated_class.append(class_i + 1)
        estimated_bbox = torch.stack(estimated_bbox).to(device)
        estimated_class= torch.tensor(estimated_class).to(device)
    
        estimated_bbox = normalize_bbox(estimated_bbox, image_size_i)
        Pseudo_labels.append({'boxes':estimated_bbox, 'labels': estimated_class})
    
    return Pseudo_labels


@torch.no_grad()
def get_pseudo_label_multi_boxes(outputs, samples, targets, args, use_nms=False, use_det=False, label_filter=False):
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x[...,0], x[...,1], x[...,2], x[...,3]
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def normalize_bbox(boxes, image_size):
        h, w = image_size
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.get_device())
        return boxes
    device = samples.tensors.get_device()
    cams = outputs['cams_cls']
    cls_logits = outputs['x_logits']

    Pseudo_labels = []
    global_token_logits = outputs['x_cls_logits']
    class_token_logits = outputs['x_logits']
    # class_token_attn_logits = outputs['']
    global_token_prob = global_token_logits.sigmoid()
    class_token_prob = class_token_logits.sigmoid()
    extend_rel_token = args.extend_rel_token
    num_rel_classes = 26
    num_classes = args.num_classes
    if extend_rel_token:
        class_token_prob = class_token_prob.view(class_token_prob.shape[0], num_classes, -1).sum(dim=-1)
    image_label_probs = (global_token_prob + class_token_prob) / 2
    
    # ############
    # image_label_probs *= torch.max(outputs['x_attn_logits'].softmax(-1), dim=-1)[0]
    # image_label_probs *= torch.max(outputs['x_spat_logits'].sigmoid(), dim=-1)[0]
    # image_label_probs *= torch.max(outputs['x_cont_logits'].sigmoid(), dim=-1)[0]
    # ############

    # image_label_probs = outputs['x_cls_logits'].sigmoid()
    for batch_i in range(cams.shape[0]):
        image_size_i = samples.tensors.shape[-2:]
        image_label_i = targets[batch_i]['img_label'].data.cpu().numpy().reshape(-1)
        if label_filter:
            # thresh = min(0.05, image_label_probs[batch_i].sort(descending=True)[0][1]) # 0.2
            thresh = 0.2
            image_label_i = (image_label_probs[batch_i] >= thresh).int().cpu().numpy().reshape(-1)
        # image_score_i = cls_logits[batch_i].sigmoid().data.cpu().numpy().reshape(-1)
        # image_score_i = cls_logits[batch_i].reshape(-1)

        estimated_bbox = []
        estimated_class= []
        estimated_score = []
        for class_i in range(args.num_classes):
            if image_label_i[class_i] > 0:
                if extend_rel_token:
                    cam_i = cams[batch_i, class_i * num_rel_classes: (class_i + 1) * num_rel_classes]
                    cam_i = torch.sum(cam_i, dim=0, keepdim=True)
                else:
                    cam_i = cams[batch_i, [class_i], :, :]
                    cam_i = torch.mean(cam_i, dim=0, keepdim=True)

                cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                cam_i = resize_cam(cam_i, size=image_size_i)
                # bbox = get_bboxes(cam_i, cam_thr=args.cam_thr)

                if use_det:
                    keep = targets[batch_i]['det_labels'] == class_i + 1
                    det_boxes = targets[batch_i]['det_boxes'][keep]
                    # pdb.set_trace()
                    det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes).clamp(min=0)
                    img_h, img_w = image_size_i
                    scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(det_boxes.device)
                    det_boxes = det_boxes * scale_fct
                    det_boxes = np.array(det_boxes.cpu(), dtype=int)
                    det_scores = targets[batch_i]['det_scores'][keep]

                    # TODO nms here
                    # if len(det_boxes) > 1:
                    #     _, order = det_scores.sort(0, descending=True)
                    #     det_scores = det_scores[order]
                    #     det_boxes = torch.tensor(det_boxes, device=det_scores.device)
                    #     det_boxes = det_boxes[order]
                    #     # idx = nms(bbox, scores)
                    #     idx = soft_nms(det_boxes, det_scores, cuda=1) # TODO
                    #     det_boxes = det_boxes[idx].cpu().numpy()
                    #     det_scores = det_scores[idx]

                    det_scores = det_scores.cpu().numpy()
                    cam_i = refine_cam_with_dets(cam_i, det_boxes, det_scores)

                bbox = get_multi_bboxes(cam_i, cam_thr=args.cam_thr, area_ratio=args.multi_box_ratio)

                scores = []
                for b in bbox:
                    x0, y0, x1, y1 = b
                    truncated_cam_i = cam_i[y0:y1, x0:x1]
                    score = truncated_cam_i.mean()
                    scores.append(score)
                scores = torch.tensor(scores) 
                bbox = torch.tensor(bbox)
                
                # postprocessing like nms, soft-nms, ...
                # nms
                # if use_nms and len(bbox) > 1:
                #     _, order = scores.sort(0, descending=True)
                #     scores = scores[order]
                #     bbox = bbox[order]
                #     # idx = nms(bbox, scores)
                #     idx = soft_nms(bbox.cuda(), scores.cuda(), cuda=1) # TODO
                #     bbox = bbox[idx]
                #     scores = scores[idx]
                
                bbox = box_xyxy_to_cxcywh(bbox)
                estimated_bbox.append(bbox)
                estimated_score.append(scores)
                for _ in range(bbox.shape[0]):
                    estimated_class.append(class_i + 1)
        estimated_bbox = torch.cat(estimated_bbox, dim=0).to(device)
        estimated_score = torch.cat(estimated_score, dim=0).to(device)
        estimated_class= torch.tensor(estimated_class).to(device)

        estimated_bbox = normalize_bbox(estimated_bbox, image_size_i)
        Pseudo_labels.append({'boxes':estimated_bbox, 'labels': estimated_class, 'scores': estimated_score})
        
    
    return Pseudo_labels


@torch.no_grad()
def refine_cam_with_dets(cam, det_boxes, det_scores):
    # cam_max = cam.max()
    refined_cam = cam.copy()
    # pdb.set_trace()
    for idx in range(len(det_boxes)):
        x0, y0, x1, y1 = det_boxes[idx]
        score = det_scores[idx]
        refined_cam[y0:y1, x0:x1] += score
    # refined_cam = refined_cam.clip(0, cam_max)
    refined_cam = refined_cam / refined_cam.max()
    return refined_cam


@torch.no_grad()
def get_pseudo_label_multi_boxes_voc(outputs, samples, targets, args):
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x[...,0], x[...,1], x[...,2], x[...,3]
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def normalize_bbox(boxes, image_size):
        h, w = image_size
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.get_device())
        return boxes

    device = samples.tensors.get_device()
    cams = outputs['cams_cls']
    cls_logits = outputs['x_logits']
    Pseudo_labels = []
    for batch_i in range(cams.shape[0]):
        image_size_i = samples.tensors.shape[-2:]
        image_label_i = targets[batch_i]['label'].tolist()
        image_score_i = cls_logits[batch_i].sigmoid().data.cpu().numpy().reshape(-1)
        image_score_i = cls_logits[batch_i].reshape(-1)
        estimated_bbox = []
        estimated_class= []
        for class_i in range(args.num_classes):
            if image_label_i[class_i] > 0:
                cam_i = cams[batch_i, [class_i], :, :]
                cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                cam_i = resize_cam(cam_i, size=image_size_i)
                # bbox = get_bboxes(cam_i, cam_thr=args.cam_thr)
                bbox = get_multi_bboxes(cam_i, cam_thr=args.cam_thr)
                bbox = torch.tensor(bbox)
                bbox = box_xyxy_to_cxcywh(bbox)
                estimated_bbox.append(bbox)
                for _ in range(bbox.shape[0]):
                    estimated_class.append(class_i + 1)
        estimated_bbox = torch.cat(estimated_bbox, dim=0).to(device)
        estimated_class= torch.tensor(estimated_class).to(device)

        estimated_bbox = normalize_bbox(estimated_bbox, image_size_i)
        Pseudo_labels.append({'boxes':estimated_bbox, 'labels': estimated_class})
    
    return Pseudo_labels


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

def train_one_epoch_refine_coco(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    args=None, postprocessors=None, criterion_refine=None):
    
    model.train()
    criterion.train()
    criterion_refine.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    gettime = lambda :time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    header = f'[{gettime()}] => Epoch: [{epoch}]'
    print_freq = 100
    weight_dict = copy.deepcopy(criterion.weight_dict)
    rf_header = 'ref'
    weight_dict = get_refine_weight_dict(weight_dict, args.num_refines, header=rf_header)
    # if epoch % 20 == 0:
    #     args.bbox_loss_coef /= 10
    #     args.giou_loss_coef /= 10
    save_idx = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # real_targets = copy.deepcopy(targets)
        outputs = model(samples)
        # pseudo_label = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args)


        #################
        attn_seed_proposals = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args, use_nms=False, use_det=True)
        seed_proposals = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args, use_nms=False, use_det=False)
        
        detections = []
        for i in range(len(targets)):
            detections.append({})
            det_boxes = targets[i]['det_boxes']
            det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes).clamp(min=0)
            detections[i]['boxes'] = det_boxes
            detections[i]['labels'] = targets[i]['det_labels']
            detections[i]['scores'] = targets[i]['det_scores']

        results = []
        for i, sp in enumerate(seed_proposals):
            results.append({})
            boxes = sp['boxes']
            boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(min=0)
            results[i]['boxes'], results[i]['scores'], results[i]['labels'] = \
                weighted_boxes_fusion([boxes, detections[i]['boxes']], [sp['scores'], detections[i]['scores']], [sp['labels'], detections[i]['labels']],
                                      iou_thr=0.5)            
            results[i]['boxes'] = torch.tensor(results[i]['boxes']).float().cuda()
            # results[i]['boxes'] = box_ops.box_xyxy_to_cxcywh(results[i]['boxes']).clamp(min=0)
            results[i]['scores'] = torch.tensor(results[i]['scores']).float().cuda()
            results[i]['labels'] = torch.tensor(results[i]['labels']).int().cuda()
            # del results[i]['scores']

            img_h, img_w = orig_target_sizes[i]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=-1)
            results[i]['boxes'] = results[i]['boxes'] * scale_fct
            
            attn_seed_proposals[i]['boxes'] = box_ops.box_cxcywh_to_xyxy(attn_seed_proposals[i]['boxes']).clamp(min=0) * scale_fct
            results[i] = attention_map_wbf_fusion(attn_seed_proposals[i], results[i], scale_fct)
            
            img_size = (img_h, img_w)
            results[i]['boxes'] = normalize_bbox(results[i]['boxes'], img_size)
            results[i]['boxes'] = box_ops.box_xyxy_to_cxcywh(results[i]['boxes'])
            del results[i]['scores']
        
        loss_dict = criterion(outputs[0], targets)
        pseudo_label = results
        #################

        for t, p in zip(targets, pseudo_label):
            t.update(p)


        # ###############
        # visualize_boxes(samples.decompose()[0], outputs[0]['cams_cls'], outputs[0]['x_logits'], real_targets, save_idx, save_image=True)
        # save_idx += 1
        # # pdb.set_trace()
        # ###############


        pseudo_label_refine = get_refinements_pseudo_label(outputs, samples, targets, args, postprocessors)
        for rf, out in outputs.items():
            if rf == 0:
                continue
            loss_dict_rf = criterion_refine(out, pseudo_label_refine[rf])

            for k, v in loss_dict_rf.items():
                key = f'{rf_header}_{rf}_{k}'
                loss_dict[key] = v
        if epoch < 100: #16: # classification branch warm-up 
            for k in weight_dict:
                if not 'img_label' in k and not 'cls' in k and not 'cams' in k:
                    weight_dict[k] = 0.0

        if epoch < 100: #16:
            for k in weight_dict:
                if header in k:
                    weight_dict[k] = 0.0

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_refinements(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, refine_stage=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    nums = 250
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        outputs_from_model = model(samples)
        outputs = outputs_from_model[refine_stage]['aux_outputs'][-1]
        loss_criterion = copy.deepcopy(criterion.losses)
        criterion.losses = ['labels', 'boxes', 'cardinality']
        loss_dict = criterion(outputs, targets)
        
        weight_dict = criterion.weight_dict
        criterion.losses = loss_criterion
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        cls_dict = {'x_logits': outputs_from_model[0]['x_logits'], 
                    'x_cls_logits': outputs_from_model[0]['x_cls_logits']}
        results = postprocessors['bbox'](outputs, orig_target_sizes, cls_dict=cls_dict)
        # pdb.set_trace()
    
        
    
        for i, r in enumerate(results):
            
            pred_boxes = r['boxes']
            pred_scores= r['scores']
            pred_labels= r['labels']
            pred_classes = r['labels'].unique()
            keep_boxes = []
            keep_scores= []
            keep_labels= []
            for pc in pred_classes.tolist():
                keep_idx = (r['labels'] == pc).nonzero(as_tuple=False).reshape(-1)
                # threshold = 0.3 #TODO
                # keep_idx = torch.logical_and(r['labels'] == pc, r['scores'] >= threshold).nonzero(as_tuple=False).reshape(-1)
                cls_pred_boxes, cls_pred_score, cls_pred_labels = pred_boxes[keep_idx], pred_scores[keep_idx], pred_labels[keep_idx]
                # keep_box_idx = torchvision.ops.nms(cls_pred_boxes, cls_pred_score, iou_threshold=0.5)
                # keep_boxes.append(cls_pred_boxes[keep_box_idx])
                # keep_scores.append(cls_pred_score[keep_box_idx])
                # keep_labels.append(cls_pred_labels[keep_box_idx])

                #################
                if sum(keep_idx) > 1:
                    order = cls_pred_score.argsort(descending=True)
                    cls_pred_boxes = cls_pred_boxes[order]
                    cls_pred_score = cls_pred_score[order]
                    # idx = nms(cls_pred_boxes, cls_pred_score)
                    idx = soft_nms(cls_pred_boxes, cls_pred_score, cuda=1) # TODO
                    # idx = diou_nms(cls_pred_boxes, cls_pred_score)

                    cls_pred_boxes = cls_pred_boxes[idx]
                    cls_pred_labels = cls_pred_labels[idx]
                    cls_pred_score = cls_pred_score[idx]
                #################
                keep_boxes.append(cls_pred_boxes)
                keep_scores.append(cls_pred_score)
                keep_labels.append(cls_pred_labels)
                

            results[i]['boxes'] = torch.cat(keep_boxes)
            results[i]['scores'] = torch.cat(keep_scores)
            results[i]['labels'] = torch.cat(keep_labels)

            threshold = 0.2
            keep_idx = (results[i]['scores'] > threshold).nonzero(as_tuple=False).reshape(-1)
            results[i]['boxes'] = results[i]['boxes'][keep_idx]
            results[i]['scores'] = results[i]['scores'][keep_idx]
            results[i]['labels'] = results[i]['labels'][keep_idx]


        detections = []
        for i in range(len(targets)):
            detections.append({})
            det_boxes = targets[i]['det_boxes']
            det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes).clamp(min=0)
            detections[i]['boxes'] = det_boxes
            detections[i]['labels'] = targets[i]['det_labels']
            detections[i]['scores'] = targets[i]['det_scores']

            img_h, img_w = orig_target_sizes[i]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=-1)
            results[i]['boxes'] = results[i]['boxes'] / scale_fct
            results[i]['boxes'], results[i]['scores'], results[i]['labels'] = \
                    weighted_boxes_fusion([results[i]['boxes'], detections[i]['boxes']], [results[i]['scores'], detections[i]['scores']], [results[i]['labels'], detections[i]['labels']], \
                                        iou_thr=0.5)
            results[i]['boxes'] = torch.tensor(results[i]['boxes']).float().cuda() * scale_fct
            results[i]['scores'] = torch.tensor(results[i]['scores']).float().cuda()
            results[i]['labels'] = torch.tensor(results[i]['labels']).int().cuda() 

            
            # draw_box_to_img(data_loader.dataset._load_image(targets[i]['image_id'].item()), results[i]['boxes'], results[i]['labels'], \
            #                 results[i]['scores'], data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"])
            
            # if data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"].split('frames/')[1] == '0BH84.mp4/000395.png':
            #     pdb.set_trace()
            # else:
            #     print(data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"].split('frames/')[1])

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # pdb.set_trace()
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        # nums -= 1
        # if nums == 0:
        #     break # TODO


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=1)
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=10)
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=100)

        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=1)
        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=10)
        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=100)
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_refinements_specific_layer(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, refine_stage=0, output_layer=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs_from_model = model(samples)
        outputs = outputs_from_model[refine_stage]['aux_outputs'][output_layer]
        criterion.losses = losses = ['labels', 'boxes', 'cardinality']
        # nms_pred_boxes = torch.stack([torchvision.ops.nms(b) for b in pred_boxes], device=pred_boxes.device)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        for i, r in enumerate(results):
            pred_boxes = r['boxes']
            pred_scores= r['scores']
            pred_labels= r['labels']
            pred_classes = r['labels'].unique()
            keep_boxes = []
            keep_scores= []
            keep_labels= []
            for pc in pred_classes.tolist():
                keep_idx = (r['labels'] == pc).nonzero().reshape(-1)
                cls_pred_boxes, cls_pred_score, cls_pred_labels = pred_boxes[keep_idx], pred_scores[keep_idx], pred_labels[keep_idx]
                keep_box_idx = torchvision.ops.nms(cls_pred_boxes, cls_pred_score, iou_threshold=0.5)
                keep_boxes.append(cls_pred_boxes[keep_box_idx])
                keep_scores.append(cls_pred_score[keep_box_idx])
                keep_labels.append(cls_pred_labels[keep_box_idx])

            results[i]['boxes'] = torch.cat(keep_boxes)
            results[i]['scores'] = torch.cat(keep_scores)
            results[i]['labels'] = torch.cat(keep_labels)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_detections(model, criterion, postprocessors, data_loader, base_ds, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    # pdb.set_trace()

    # obj_dist = [0 for i in range(36)]
    # nums = 500
    recall_list = []
    iou_list = []
    all_num = 0
    to_eval = 10
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # if to_eval == 0:
        #     break
        # to_eval -= 1

        for i, target in enumerate(targets):
            det_boxes = target['det_boxes']
            det_labels = target['det_labels']
            det_scores = target['det_scores']

            det_boxes = torchvision.ops.box_convert(det_boxes, 'cxcywh', 'xyxy')
            tgt_boxes = torchvision.ops.box_convert(target['boxes'], 'cxcywh', 'xyxy')
            iou = torchvision.ops.box_iou(tgt_boxes, det_boxes)
            all_num += len(target['labels'])

            for lb_i, label in enumerate(target['labels']):
                mask = label == det_labels
                if mask.any():
                    max_iou = max(iou[lb_i, mask]).item()
                    iou_list.append(max_iou)

        continue
        # samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = []
        for i in range(len(targets)):
            results.append({})
            det_boxes = targets[i]['det_boxes']
            det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes).clamp(min=0)
            img_h, img_w = orig_target_sizes[i]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=-1)
            det_boxes = det_boxes * scale_fct
            results[i]['boxes'] = det_boxes
            results[i]['labels'] = targets[i]['det_labels']
            results[i]['scores'] = targets[i]['det_scores']

            # ########### NMS TODO
            # boxes_list = []
            # labels_list = []
            # scores_list = []
            # for class_i in range(args.num_classes):
            #     cls_idx = results[i]['labels'] == class_i
            #     if sum(cls_idx) == 0:
            #         continue
            #     boxes = results[i]['boxes'][cls_idx]
            #     labels = results[i]['labels'][cls_idx]
            #     scores = results[i]['scores'][cls_idx]
            #     ###########
            #     # if sum(cls_idx) > 1:
            #     #     order = scores.argsort(descending=True)
            #     #     boxes = boxes[order]
            #     #     scores = scores[order]
            #     #     # idx = nms(boxes, scores)
            #     #     idx = soft_nms(boxes, scores, cuda=1) # TODO
            #     #     boxes = boxes[idx]
            #     #     labels = labels[idx]
            #     #     scores = scores[idx]
            #     #############
            #     boxes_list.append(boxes)
            #     labels_list.append(labels)
            #     scores_list.append(scores)
            # results[i]['boxes'] = torch.cat(boxes_list).to(det_boxes)
            # results[i]['labels'] = torch.cat(labels_list).to(det_boxes)
            # results[i]['scores'] = torch.cat(scores_list).to(det_boxes)
            # ############## 

            # draw_box_to_img(data_loader.dataset._load_image(targets[i]['image_id'].item()), results[i]['boxes'], results[i]['labels'], \
            #                 results[i]['scores'], data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"])
            
            
            # tgt = {}
            # tgt['boxes'] = targets[i]['boxes'] 
            # tgt['boxes'][:, :2] -= tgt['boxes'][:, 2:] / 2
            # tgt['boxes'][:, 2:] += tgt['boxes'][:, :2]
            # tgt['boxes'] *= scale_fct
            # tgt['labels'] = targets[i]['labels']
            # tgt['scores'] = torch.ones_like(tgt['labels'])
            # draw_box_to_img(data_loader.dataset._load_image(targets[i]['image_id'].item()), tgt['boxes'], tgt['labels'], \
            #                 tgt['scores'], data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"])
            
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # nums -= 1
        # if nums == 0:
        #     break # TODO

    print('recall ratio = ', len(iou_list) / all_num)
    print('miou = ', np.mean(iou_list))
    pdb.set_trace()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=1)
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=10)
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=100)

        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=1)
        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=10)
        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=100)
    
    return None, coco_evaluator

@torch.no_grad()
def evaluate_seed_proposal(model, criterion, postprocessors, data_loader, base_ds, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    to_eval = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # if to_eval == 250:
        #     break
        # to_eval += 1
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        seed_proposals = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args, use_nms=False)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = []
        for i, sp in enumerate(seed_proposals):
            results.append({})
            boxes = sp['boxes']
            # pdb.set_trace()
            boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(min=0)
            img_h, img_w = orig_target_sizes[i]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=-1)
            boxes = boxes * scale_fct
            results[i]['boxes'] = boxes
            results[i]['labels'] = sp['labels']
            results[i]['scores'] = sp['scores']
            # results[i]['scores'] = torch.ones_like(sp['labels'])

            # draw_box_to_img(data_loader.dataset._load_image(targets[i]['image_id'].item()), results[i]['boxes'], results[i]['labels'], \
            #                 results[i]['scores'], data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"])

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)

     # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=1)
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=10)
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=100)

        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=1)
        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=10)
        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=100)
    
    return None, coco_evaluator

@torch.no_grad()
def evaluate_seed_proposal_detections(model, criterion, postprocessors, data_loader, base_ds, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    to_eval = 0
    sum_obj = 0
    hit_attn_rels = 0
    hit_spat_rels = 0
    hit_cont_rels = 0
    iou_list = []
    all_num = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # if to_eval == 100:
        #     break
        # to_eval += 1
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        detections = []
        for i in range(len(targets)):
            detections.append({})
            det_boxes = targets[i]['det_boxes']
            det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes).clamp(min=0)
            detections[i]['boxes'] = det_boxes
            detections[i]['labels'] = targets[i]['det_labels']
            detections[i]['scores'] = targets[i]['det_scores']

        outputs = model(samples)

        # for i, target_i in enumerate(targets):
        #     for obj_idx, label in enumerate(target_i['labels']):
        #         if label == 1:
        #             continue
        #         pred_attn_label = outputs[0]['x_attn_logits'][i][label - 1].argmax()
        #         pred_spat_label = outputs[0]['x_spat_logits'][i][label - 1].argmax()
        #         pred_cont_label = outputs[0]['x_cont_logits'][i][label - 1].argmax()
        #         attn_label = torch.where(target_i['attention_label'][obj_idx])[0]
        #         spat_label = torch.where(target_i['spatial_label'][obj_idx])[0]
        #         cont_label = torch.where(target_i['contact_label'][obj_idx])[0]
        #         sum_obj += 1
        #         hit_attn_rels += int(pred_attn_label in attn_label)
        #         hit_spat_rels += int(pred_spat_label in spat_label)
        #         hit_cont_rels += int(pred_cont_label in cont_label)


        # #################### visualization
        # cams = outputs[0]['cams_cls']
        # for i in range(len(cams)):
        #     cams_i = cams[i]
        #     tgt_label = targets[i]['labels']
        #     for lb in tgt_label:
        #         cam_i = cams_i[lb - 1]
        #         cam_i = cam_i.detach().cpu().numpy()
        #         plot_attn_weight(cam_i, data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"], lb)
        # continue


        attn_seed_proposals = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args, use_nms=False, use_det=True, label_filter=True)
        seed_proposals = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args, use_nms=False, use_det=False, label_filter=True)
        results = []
        for i, sp in enumerate(seed_proposals):
            results.append({})
            boxes = sp['boxes']
            boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(min=0)
            results[i]['boxes'], results[i]['scores'], results[i]['labels'] = \
                weighted_boxes_fusion([boxes, detections[i]['boxes']], [sp['scores'], detections[i]['scores']], [sp['labels'], detections[i]['labels']], \
                                      iou_thr=0.5)
            results[i]['boxes'] = torch.tensor(results[i]['boxes']).float().cuda()
            results[i]['scores'] = torch.tensor(results[i]['scores']).float().cuda()
            results[i]['labels'] = torch.tensor(results[i]['labels']).int().cuda()

            # ################
            # results[i]['boxes'] = torch.cat([boxes, detections[i]['boxes']])
            # results[i]['labels'] = torch.cat([sp['labels'], detections[i]['labels']])
            # results[i]['scores'] = torch.cat([sp['scores'], detections[i]['scores']])
            # ################

            img_h, img_w = orig_target_sizes[i]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=-1)
            results[i]['boxes'] = results[i]['boxes'] * scale_fct

            attn_seed_proposals[i]['boxes'] = box_ops.box_cxcywh_to_xyxy(attn_seed_proposals[i]['boxes']).clamp(min=0) * scale_fct

            # ################
            # pred_boxes = results[i]['boxes']
            # pred_scores= results[i]['scores']
            # pred_labels= results[i]['labels']
            # pred_classes = pred_labels.unique()
            # keep_boxes = []
            # keep_scores= []
            # keep_labels= []
            # for pc in pred_classes.tolist():
            #     keep_idx = (pred_labels == pc).nonzero(as_tuple=False).reshape(-1)
            #     cls_pred_boxes, cls_pred_score, cls_pred_labels = pred_boxes[keep_idx], pred_scores[keep_idx], pred_labels[keep_idx]
            #     if sum(keep_idx) > 1:
            #         order = cls_pred_score.argsort(descending=True)
            #         cls_pred_boxes = cls_pred_boxes[order]
            #         cls_pred_score = cls_pred_score[order]
            #         # idx = nms(cls_pred_boxes, cls_pred_score)
            #         idx = soft_nms(cls_pred_boxes, cls_pred_score, cuda=1) # TODO
            #         # idx = diou_nms(cls_pred_boxes, cls_pred_score)

            #         cls_pred_boxes = cls_pred_boxes[idx]
            #         cls_pred_labels = cls_pred_labels[idx]
            #         cls_pred_score = cls_pred_score[idx]
            #     keep_boxes.append(cls_pred_boxes)
            #     keep_scores.append(cls_pred_score)
            #     keep_labels.append(cls_pred_labels)

            # results[i]['boxes'] = torch.cat(keep_boxes)
            # results[i]['scores'] = torch.cat(keep_scores)
            # results[i]['labels'] = torch.cat(keep_labels)
            # ################

            # draw_box_to_img(data_loader.dataset._load_image(targets[i]['image_id'].item()), results[i]['boxes'], results[i]['labels'], \
            #                 results[i]['scores'], data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"])
            

            # TODO
            results[i] = attention_map_wbf_fusion(attn_seed_proposals[i], results[i], scale_fct) 
            # results[i]['boxes'] /= scale_fct

            # results[i] = attn_seed_proposals[i]
            # results[i] = sp
            # results[i]['boxes'] = box_ops.box_cxcywh_to_xyxy(sp['boxes']).clamp(min=0) * scale_fct
            # results[i]['boxes'] *= scale_fct

        # for i, target in enumerate(targets):
        #     det_boxes = results[i]['boxes']
        #     det_labels = results[i]['labels']
        #     det_scores = results[i]['scores']
        #     tgt_boxes = torchvision.ops.box_convert(target['boxes'], 'cxcywh', 'xyxy')
        #     iou = torchvision.ops.box_iou(tgt_boxes, det_boxes)
        #     all_num += len(target['labels'])

        #     for lb_i, label in enumerate(target['labels']):
        #         mask = label == det_labels
        #         if mask.any():
        #             max_iou = max(iou[lb_i, mask]).item()
        #             iou_list.append(max_iou)

        # continue
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # pdb.set_trace()
    # print('class recall = ', len(iou_list) / all_num)
    # print('miou = ', np.mean(iou_list))
    # pdb.set_trace()
     # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=1)
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=10)
        coco_evaluator.coco_eval['bbox']._summarize(ap=1, iouThr=0.5, areaRng='all', maxDets=100)

        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=1)
        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=10)
        coco_evaluator.coco_eval['bbox']._summarize(ap=0, iouThr=0.5, areaRng='all', maxDets=100)

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    
    # hit_rels = hit_attn_rels + hit_spat_rels + hit_cont_rels
    # attn_accuracy = hit_attn_rels / sum_obj
    # spat_accuracy = hit_spat_rels / sum_obj
    # cont_accuracy = hit_cont_rels / sum_obj
    # total_accuracy = hit_rels / (sum_obj * 3)
    # stats['rel_accuracy'] = [attn_accuracy, spat_accuracy, cont_accuracy, total_accuracy]

    # print('attention accuracy:', attn_accuracy)
    # print('spatial accuracy:', spat_accuracy)
    # print('contacting accuracy:', cont_accuracy)
    # print('total accuracy:', total_accuracy)
    # print('nums:{} + {} + {} = {} / {}'.format(hit_attn_rels, hit_spat_rels, hit_cont_rels, hit_rels, sum_obj * 3))

    return stats, coco_evaluator


@torch.no_grad()
def evaluate_save_det(model, criterion, postprocessors, data_loader, device, output_dir, args, refine_stage=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    roi_align_ops = RoIAlign(output_size=(7, 7), spatial_scale= 1.0 / 16, sampling_ratio=2)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):

        need_save = False
        for i in range(len(targets)):
            vid, frame = data_loader.dataset.coco.imgs[targets[i]['image_id'].item()]['file_name'].split('/')[1:]
            save_root = os.path.join('/home/wangguan/SPE-master/data/wk_detection_results_full005_ckpt15', vid, frame)
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            det_path = os.path.join(save_root, 'dets.npy')
            feat_path = os.path.join(save_root, 'feat.npy')
            if not os.path.exists(det_path) or not os.path.exists(feat_path):
                need_save = True
        if not need_save:
            continue



        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # outputs_from_model, memory = model(samples, return_memory=True)
        # outputs = outputs_from_model[refine_stage]['aux_outputs'][-1]


        # cls_dict = {'x_logits': outputs_from_model[0]['x_logits'], 
        #             'x_cls_logits': outputs_from_model[0]['x_cls_logits']}
        # results = postprocessors['bbox'](outputs, orig_target_sizes, need_scale=False, cls_dict=cls_dict) #TODO



        ########################################
        detections = []
        for i in range(len(targets)):
            detections.append({})
            det_boxes = targets[i]['det_boxes']
            det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes).clamp(min=0)
            detections[i]['boxes'] = det_boxes
            detections[i]['labels'] = targets[i]['det_labels']
            detections[i]['scores'] = targets[i]['det_scores']

        outputs, memory = model(samples, return_memory=True)
        # detection -> attention map -> seed proposals
        attn_seed_proposals = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args, use_nms=False, use_det=True, label_filter=True)
        # attention map -> seed proposals
        seed_proposals = get_pseudo_label_multi_boxes(outputs[0], samples, targets, args, use_nms=False, use_det=False, label_filter=True)
        
        results = []
        for i, sp in enumerate(seed_proposals):
            
            results.append({})

            vid, frame = data_loader.dataset.coco.imgs[targets[i]['image_id'].item()]['file_name'].split('/')[1:]
            save_root = os.path.join('/home/wangguan/SPE-master/data/wk_detection_results_full005_ckpt15', vid, frame)
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            det_path = os.path.join(save_root, 'dets.npy')
            feat_path = os.path.join(save_root, 'feat.npy')
            if os.path.exists(det_path) and os.path.exists(feat_path):
                continue
        
            boxes = sp['boxes']
            boxes = box_ops.box_cxcywh_to_xyxy(boxes).clamp(min=0)

            # attention -> seed proposals -> detection
            results[i]['boxes'], results[i]['scores'], results[i]['labels'] = \
                weighted_boxes_fusion([boxes, detections[i]['boxes']], [sp['scores'], detections[i]['scores']], [sp['labels'], detections[i]['labels']], \
                                      iou_thr=0.5)
            results[i]['boxes'] = torch.tensor(results[i]['boxes']).cuda()
            results[i]['scores'] = torch.tensor(results[i]['scores']).cuda()
            results[i]['labels'] = torch.tensor(results[i]['labels']).cuda()
            # keep_boxes_ts = results[i]['boxes']

            img_h, img_w = orig_target_sizes[i]
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=-1)
            results[i]['boxes'] = results[i]['boxes'] * scale_fct

            ## multi-step fusion
            attn_seed_proposals[i]['boxes'] = box_ops.box_cxcywh_to_xyxy(attn_seed_proposals[i]['boxes']).clamp(min=0) * scale_fct
            results[i] = attention_map_wbf_fusion(attn_seed_proposals[i], results[i], scale_fct)

            # # only one-step fusion
            # # results[i] = attn_seed_proposals[i]
            # results[i] = sp
            # results[i]['boxes'] = box_ops.box_cxcywh_to_xyxy(results[i]['boxes']).clamp(min=0) * scale_fct

            keep_boxes_ts = results[i]['boxes'] / scale_fct

            # obtain the roi feature according to the boxes
            bs, N, mapped_h, mapped_w = samples.tensors.shape
            mapped_fct = torch.tensor([mapped_w, mapped_h, mapped_w, mapped_h]).to(keep_boxes_ts.device)
            mapped_boxes = keep_boxes_ts * mapped_fct
            mapped_boxes = torch.cat([torch.zeros(len(mapped_boxes), 1).to(mapped_boxes), mapped_boxes], dim=-1)
            memory_h, memory_w = outputs[0]['cams_cls'].shape[2:]
            memory_mat = memory[:, i, :].permute(1, 0).reshape(1, memory.shape[-1], memory_h, memory_w)
            roi_features = roi_align_ops(memory_mat, mapped_boxes.float()).mean(dim=[2,3]).cpu().numpy()

            save_det_list = []
            for ridx in range(len(results[i]['boxes'])):
                det_dict = {}
                det_dict['rect'] = results[i]['boxes'][ridx].cpu().tolist()
                det_dict['conf'] = results[i]['scores'][ridx].cpu().tolist()
                det_dict['class'] = results[i]['labels'][ridx].cpu().int().tolist()
                save_det_list.append(det_dict)

            # pdb.set_trace()
            np.save(det_path, save_det_list)
            np.save(feat_path, roi_features)
        ###############################


        # detections = []
        # for i in range(len(targets)):
        #     detections.append({})
        #     det_boxes = targets[i]['det_boxes']
        #     det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes).clamp(min=0)
        #     detections[i]['boxes'] = det_boxes
        #     detections[i]['labels'] = targets[i]['det_labels']
        #     detections[i]['scores'] = targets[i]['det_scores']


        # for i, r in enumerate(results):
        #     pred_boxes = r['boxes']
        #     pred_scores= r['scores']
        #     pred_labels= r['labels']
        #     pred_classes = r['labels'].unique()
        #     keep_boxes = []
        #     keep_scores= []
        #     keep_labels= []
        #     img_h, img_w = orig_target_sizes[i]
        #     scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(pred_boxes.device)
        #     for pc in pred_classes.tolist():
        #         keep_idx = (r['labels'] == pc).nonzero(as_tuple=False).reshape(-1)
        #         cls_pred_boxes, cls_pred_score, cls_pred_labels = pred_boxes[keep_idx], pred_scores[keep_idx], pred_labels[keep_idx]
        #         # keep_box_idx = torchvision.ops.nms(cls_pred_boxes, cls_pred_score, iou_threshold=0.5)
        #         # keep_boxes.append(cls_pred_boxes[keep_box_idx])
        #         # keep_scores.append(cls_pred_score[keep_box_idx])
        #         # keep_labels.append(cls_pred_labels[keep_box_idx])

                
        #         ##################
        #         if sum(keep_idx) > 1:
        #             order = cls_pred_score.argsort(descending=True)
        #             cls_pred_boxes = cls_pred_boxes[order]
        #             cls_pred_score = cls_pred_score[order]
        #             # idx = nms(cls_pred_boxes, cls_pred_score)
        #             scaled_pred_boxes = cls_pred_boxes * scale_fct
        #             idx = soft_nms(scaled_pred_boxes, cls_pred_score, cuda=1) # TODO
        #             # idx = diou_nms(cls_pred_boxes, cls_pred_score)

        #             cls_pred_boxes = cls_pred_boxes[idx]
        #             cls_pred_labels = cls_pred_labels[idx]
        #             cls_pred_score = cls_pred_score[idx]
        #         #################
        #         keep_boxes.append(cls_pred_boxes)
        #         keep_scores.append(cls_pred_score)
        #         keep_labels.append(cls_pred_labels)

        #     keep_boxes_ts = torch.cat(keep_boxes)
        #     results[i]['boxes'] = keep_boxes_ts #* scale_fct
        #     results[i]['scores'] = torch.cat(keep_scores)
        #     results[i]['labels'] = torch.cat(keep_labels)

        #     threshold = 0.2
        #     keep_idx = (results[i]['scores'] > threshold).nonzero(as_tuple=False).reshape(-1)
        #     results[i]['boxes'] = results[i]['boxes'][keep_idx]
        #     results[i]['scores'] = results[i]['scores'][keep_idx]
        #     results[i]['labels'] = results[i]['labels'][keep_idx]
        #     keep_boxes_ts = results[i]['boxes']

        #     ###############
        #     results[i]['boxes'], results[i]['scores'], results[i]['labels'] = \
        #             weighted_boxes_fusion([results[i]['boxes'], detections[i]['boxes']], [results[i]['scores'], detections[i]['scores']], [results[i]['labels'], detections[i]['labels']], \
        #                                 iou_thr=0.5)
        #     results[i]['boxes'] = torch.tensor(results[i]['boxes']).float().cuda()
        #     keep_boxes_ts = results[i]['boxes']
        #     results[i]['boxes'] *= scale_fct
        #     results[i]['scores'] = torch.tensor(results[i]['scores']).float().cuda()
        #     results[i]['labels'] = torch.tensor(results[i]['labels']).int().cuda()
        #     ###############

            
        #     # if data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"].split('frames/')[1] == '0BH84.mp4/000395.png':
        #     #     pdb.set_trace()
        #     # else:
        #     #     print(data_loader.dataset.coco.loadImgs(targets[i]['image_id'].item())[0]["file_name"].split('frames/')[1])
        #     #     continue

        #     # obtain the roi feature according to the boxes
        #     bs, N, mapped_h, mapped_w = samples.tensors.shape
        #     mapped_fct = torch.tensor([mapped_w, mapped_h, mapped_w, mapped_h]).to(keep_boxes_ts.device)
        #     mapped_boxes = keep_boxes_ts * mapped_fct
        #     mapped_boxes = torch.cat([torch.zeros(len(mapped_boxes), 1).to(mapped_boxes), mapped_boxes], dim=-1)
        #     memory_h, memory_w = outputs_from_model[refine_stage]['cams_cls'].shape[2:]
        #     memory_mat = memory[:, i, :].permute(1, 0).reshape(1, memory.shape[-1], memory_h, memory_w)
        #     roi_features = roi_align_ops(memory_mat, mapped_boxes).mean(dim=[2,3]).cpu().numpy()

        #     save_det_list = []
        #     for ridx in range(len(results[i]['boxes'])):
        #         det_dict = {}
        #         det_dict['rect'] = results[i]['boxes'][ridx].cpu().tolist()
        #         det_dict['conf'] = results[i]['scores'][ridx].cpu().tolist()
        #         det_dict['class'] = results[i]['labels'][ridx].cpu().tolist()
        #         save_det_list.append(det_dict)

        #     vid, frame = data_loader.dataset.coco.imgs[targets[i]['image_id'].item()]['file_name'].split('/')[1:]
        #     save_root = os.path.join('/home/wangguan/SPE-master/data/wk_detection_results_weighted', vid, frame)
        #     if not os.path.exists(save_root):
        #         os.makedirs(save_root)
        #     det_path = os.path.join(save_root, 'dets.npy')
        #     feat_path = os.path.join(save_root, 'feat.npy')
        #     if os.path.exists(det_path) and os.path.exists(feat_path):
        #         continue
        #     # pdb.set_trace()
        #     np.save(det_path, save_det_list)
        #     np.save(feat_path, roi_features)


@torch.no_grad()
def save_img_label(model, criterion, postprocessors, data_loader, device, output_dir, refine_stage=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    img_root = '/home/wangguan/SPE-master/data/frames'
    img2label = {}
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        for target in targets:
            img_id = target['image_id'].item()
            vid, frame = data_loader.dataset.coco.imgs[img_id]['file_name'].split('/')[1:]
            frame_path = os.path.join(img_root, vid, frame)
            img2label[frame_path] = (torch.where(target['img_label'])[0] + 1).tolist()

    save_path = '/home/wangguan/SPE-master/data/annotations/weak/wk_train_label.json'
    # with open(save_path, 'w') as f:
    #     json.dump(img2label, f)
    pdb.set_trace()
    print()


def draw_box_to_img(img, boxes, labels, scores, name, ext=False):
    
    OBJ_CLASSES = ("background", "person", "bag", "bed", "blanket", "book", "box", "broom", "chair", 
           "closet/cabinet", "clothes", "cup", "dish", "door", 
           "doorknob", "doorway", "floor", "food", "groceries", "laptop", 
           "light", "medicine", "mirror", "paper", "phone", 
           "picture", "pillow", "refrigerator", "sandwich", "shelf", "shoe", "sofa",
           "table", "television", "towel", "vacuum", "window")
    
    base_img = img.copy()
    img_name = name.split('frames/')[1].replace('/', '_')
    draw = ImageDraw.Draw(img)
    pid = labels > 0 # == 1
    pboxes = boxes[pid]
    pscores = scores[pid]
    nump = len(pboxes)

    img_idx = 0
    for i in range(nump):
        box = pboxes[i].cpu().tolist()
        score = str(pscores[i].item())
        label_score = OBJ_CLASSES[labels[i]] + "_" + score
        draw.rectangle(box, outline="red")  
        draw.text(box[:2], label_score, fill="white")
        folder = os.path.join("/home/wangguan/SPE-master/visualization/comparison/", img_name, 'proposal')
        # folder = os.path.join("/home/wangguan/SPE-master/visualization/detections/", img_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = os.path.join(folder, str(img_idx) + '.jpg')
        img.save(save_path)
        if i % 5 == 0 and i > 0:
            img = base_img.copy()
            draw = ImageDraw.Draw(img)
            img_idx += 1
            

    # pdb.set_trace()
    # print()


@torch.no_grad()
def evaluate_save_feat_by_ext_det(model, criterion, postprocessors, data_loader, device, output_dir, refine_stage=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    roi_align_ops = RoIAlign(output_size=(7, 7), spatial_scale= 1.0 / 16, sampling_ratio=2)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):

        
        # if os.path.exists(det_path) and os.path.exists(feat_path):
        #     continue

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs_from_model, memory = model(samples, return_memory=True)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = []
        for i in range(len(targets)):
            results.append({})
            det_boxes = targets[i]['det_boxes']
            det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes).clamp(min=0)
            img_h, img_w = orig_target_sizes[i]
            # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=-1)
            # det_boxes = det_boxes * scale_fct
            results[i]['boxes'] = det_boxes
            results[i]['labels'] = targets[i]['det_labels']
            results[i]['scores'] = targets[i]['det_scores']

            # obtain the roi feature according to the boxes
            bs, N, mapped_h, mapped_w = samples.tensors.shape
            mapped_fct = torch.tensor([mapped_w, mapped_h, mapped_w, mapped_h]).to(memory.device)

            mapped_boxes = results[i]['boxes'] * mapped_fct
            mapped_boxes = torch.cat([torch.zeros(len(mapped_boxes), 1).to(mapped_boxes), mapped_boxes], dim=-1)
            memory_h, memory_w = outputs_from_model[refine_stage]['cams_cls'].shape[2:]
            memory_mat = memory[:, i, :].permute(1, 0).reshape(1, memory.shape[-1], memory_h, memory_w)
            roi_features = roi_align_ops(memory_mat, mapped_boxes).mean(dim=[2,3]).cpu().numpy()

            save_det_list = []
            results[i]['boxes'] *= torch.tensor([img_w, img_h, img_w, img_h]).to(results[i]['boxes'].device)
            # pdb.set_trace()
            for ridx in range(len(results[i]['boxes'])):
                det_dict = {}
                det_dict['rect'] = results[i]['boxes'][ridx].cpu().tolist()
                det_dict['conf'] = results[i]['scores'][ridx].cpu().tolist()
                det_dict['class'] = results[i]['labels'][ridx].cpu().tolist()
                save_det_list.append(det_dict)

            vid, frame = data_loader.dataset.coco.imgs[targets[i]['image_id'].item()]['file_name'].split('/')[1:]
            save_root = os.path.join('/home/wangguan/SPE-master/data/ext_detection_results', vid, frame)
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            det_path = os.path.join(save_root, 'dets.npy')
            feat_path = os.path.join(save_root, 'feat.npy')
            if os.path.exists(det_path) and os.path.exists(feat_path):
                continue
            np.save(det_path, save_det_list)
            np.save(feat_path, roi_features)


def attention_map_wbf_fusion(attn_seed_proposals, wbf_seed_proposals, scale_fct):
    
    # idxes = attn_seed_proposals['scores'].argsort(descending=True)

    # maxIdx = idxes#[:5]
    # wbf_seed_proposals['boxes'] = torch.cat([wbf_seed_proposals['boxes'], attn_seed_proposals['boxes'][maxIdx]])
    # wbf_seed_proposals['labels'] = torch.cat([wbf_seed_proposals['labels'], attn_seed_proposals['labels'][maxIdx]])
    # wbf_seed_proposals['scores'] = torch.cat([wbf_seed_proposals['scores'], attn_seed_proposals['scores'][maxIdx]])
    # pdb.set_trace()

    # pred_boxes = wbf_seed_proposals['boxes']
    # pred_scores= wbf_seed_proposals['scores']
    # pred_labels= wbf_seed_proposals['labels']
    # pred_classes = pred_labels.unique()
    # keep_boxes = []
    # keep_scores= []
    # keep_labels= []
    # for pc in pred_classes.tolist():
    #     keep_idx = (pred_labels == pc).nonzero(as_tuple=False).reshape(-1)
    #     cls_pred_boxes, cls_pred_score, cls_pred_labels = pred_boxes[keep_idx], pred_scores[keep_idx], pred_labels[keep_idx]
    #     if sum(keep_idx) > 1:
    #         order = cls_pred_score.argsort(descending=True)
    #         cls_pred_boxes = cls_pred_boxes[order]
    #         cls_pred_score = cls_pred_score[order]
    #         # idx = nms(cls_pred_boxes, cls_pred_score)
    #         idx = soft_nms(cls_pred_boxes, cls_pred_score, cuda=1) # TODO
    #         # idx = diou_nms(cls_pred_boxes, cls_pred_score)

    #         cls_pred_boxes = cls_pred_boxes[idx]
    #         cls_pred_labels = cls_pred_labels[idx]
    #         cls_pred_score = cls_pred_score[idx]
    #     keep_boxes.append(cls_pred_boxes)
    #     keep_scores.append(cls_pred_score)
    #     keep_labels.append(cls_pred_labels)

    # wbf_seed_proposals['boxes'] = torch.cat(keep_boxes)
    # wbf_seed_proposals['scores'] = torch.cat(keep_scores)
    # wbf_seed_proposals['labels'] = torch.cat(keep_labels)
    attn_seed_proposals['boxes'] /= scale_fct
    wbf_seed_proposals['boxes'] /= scale_fct
    wbf_seed_proposals['boxes'], wbf_seed_proposals['scores'], wbf_seed_proposals['labels'] = \
        weighted_boxes_fusion([wbf_seed_proposals['boxes'], attn_seed_proposals['boxes']], 
                              [wbf_seed_proposals['scores'], attn_seed_proposals['scores']], 
                              [wbf_seed_proposals['labels'], attn_seed_proposals['labels']],
                              iou_thr=0.5)
    wbf_seed_proposals['boxes'] = torch.tensor(wbf_seed_proposals['boxes']).float().cuda() * scale_fct
    wbf_seed_proposals['scores'] = torch.tensor(wbf_seed_proposals['scores']).float().cuda()
    wbf_seed_proposals['labels'] = torch.tensor(wbf_seed_proposals['labels']).long().cuda()

    return wbf_seed_proposals


def plot_attn_weight(attention_mask, img_path, label):

    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    OBJ_CLASSES = ("background", "person", "bag", "bed", "blanket", "book", "box", "broom", "chair", 
           "closet/cabinet", "clothes", "cup", "dish", "door", 
           "doorknob", "doorway", "floor", "food", "groceries", "laptop", 
           "light", "medicine", "mirror", "paper", "phone", 
           "picture", "pillow", "refrigerator", "sandwich", "shelf", "shoe", "sofa",
           "table", "television", "towel", "vacuum", "window")
    
    img_name = img_path.split('/')[1].split('.')[0] + '_' + img_path.split('/')[2].split('.')[0]
    img_path = './data/' + img_path
    print("load image from: ", img_path)
    # img = Image.open(img_path, mode='r')
    # img_w, img_h = img.size[0], img.size[1]
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_w, 0.02 * img_h))

    # scale the image
    # ratio = 1
    # pdb.set_trace()
    # img_w, img_h = int(img.size[0] * ratio), int(img.size[1] * ratio)
    # img = img.resize((img_h, img_w))

    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    # attention_mask = 
    mask = cv2.resize(attention_mask, (img_w, img_h))
    # normed_mask = mask / mask.max()
    normed_mask = (mask - mask.min()) / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest')

    # build save path
    save_folder = os.path.join('./visualization/', img_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, OBJ_CLASSES[label]+'.jpg')
    ori_path = os.path.join(save_folder, 'orig.jpg')
    
    # pre-process and save image
    print("save image to: " + save_path)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    quality = 200
    plt.savefig(save_path, dpi=quality)

    if not os.path.exists(ori_path):
        # save original image file
        cv2.imwrite(ori_path, img)
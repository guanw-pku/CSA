# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import pdb
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, evaluate_refinements, train_one_epoch, train_one_epoch_refine_coco, evaluate_refinements_specific_layer, \
    evaluate_detections, evaluate_seed_proposal, evaluate_seed_proposal_detections, evaluate_save_det, save_img_label, evaluate_save_feat_by_ext_det
from models import build_model
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_cls_head', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--cam_thr', default=0.2, type=float, 
                        help='threshold of generating bounding box from attention maps')
    parser.add_argument('--multi_box_ratio', default=0.5, type=float)
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='TSCAM_cait_XXS24', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone_drop_rate', default=0., type=float)
    parser.add_argument('--drop_path_rate', default=0., type=float)
    parser.add_argument('--drop_block_rate', default=0., type=float)
    parser.add_argument('--drop_attn_rate', default=0., type=float)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--layer_to_det', default=23, type=int, 
                        help='feature layer feed to encoder-decoder')
    parser.add_argument('--num_refines', default=1, type=int, 
                        help='Number of refinements')
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--hung_match_ratio', default=5, type=int,
                        help="match ratio of hungarian algorithm")
    parser.add_argument('--hungarian_multi', action='store_true')
    parser.add_argument('--box_jitter', default=0.1, type=float,
                        help="box jitter during box repeating")
    parser.add_argument('--drloc', action='store_true',
                        help="drloc module for data efficient training")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=2, type=float)
    parser.add_argument('--img_label_loss_coef', default=1, type=float)
    parser.add_argument('--img_label_tokens_loss_coef', default=1, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--drloc_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=2, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--max_size', default=448, type=int)
    parser.add_argument('--fixed_size', action='store_true')
    parser.add_argument('--area_ratio', default=0.5, type=float)
    

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # import external object detection results
    parser.add_argument('--ext_det', action='store_true') # utilize external detection results
    parser.add_argument('--infer_save', action='store_true') # preprocess detection results with trained model
    parser.add_argument('--infer_data', default='test', choices=['train', 'test', 'total'])
    
    parser.add_argument('--extend_rel_token', action='store_true') # num_classes tokens -> num_classes * num_rel_classes tokens
    parser.add_argument('--pair_attn_cls', action='store_true') # use pair attention maps to classify relations
    parser.add_argument('--rel_loss_coeff', default=0.1, type=float)
    parser.add_argument('--add_relation_token', action='store_true') # freeze object classification params, only new relation tokens are to be trained
    parser.add_argument('--full_ratio', default=0.0, type=float) # ratio of full supervision training data to use
    return parser



def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    assert int(args.pair_attn_cls) + int(args.extend_rel_token) + int(args.add_relation_token) <= 1

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, criterion_refine, postprocessors, refine_postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad and 'blocks_token_only' not in n],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad and 'blocks_token_only' in n],
            "lr": args.lr_cls_head,
        },

    ]

    if args.add_relation_token:
        param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if 'extra_rel' in n and p.requires_grad]}]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', dataset_file=args.dataset_file, args=args)
    dataset_val = build_dataset(image_set='val', dataset_file=args.dataset_file, args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.add_relation_token and not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            
    if args.eval:
        if not args.infer_save:
            test_stats, coco_evaluator = evaluate_seed_proposal_detections(model, criterion, postprocessors,
                                        data_loader_val, base_ds, device, args)
        else:
            evaluate_save_det(model, criterion, postprocessors, data_loader_val, device, output_dir, args)
            
        # test_stats, coco_evaluator = evaluate_refinements(model, criterion, postprocessors,
                                                # data_loader_val, base_ds, device, args.output_dir, refine_stage=0)
        
        # test_stats, coco_evaluator = evaluate_detections(model, criterion, postprocessors,
        #                                         data_loader_val, base_ds, device, args)
        
        # test_stats, coco_evaluator = evaluate_seed_proposal(model, criterion, postprocessors,
        #                                         data_loader_val, base_ds, device, args)

        # save_img_label(model, criterion, postprocessors,data_loader_val, device, output_dir)

        # evaluate_save_feat_by_ext_det(model, criterion, postprocessors, data_loader_val, device, output_dir, refine_stage=0)

        return

    print("Start training")
    start_time = time.time()
    ap50_max = 0
    save_flag = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch_refine_coco(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args=args, criterion_refine=criterion_refine, postprocessors=refine_postprocessors)
        lr_scheduler.step()
        test_stats = None
        coco_evaluator = None

        if (epoch + 1) % 2 == 0: 
            test_stats, coco_evaluator = evaluate_seed_proposal_detections(model, criterion, postprocessors,
                                                        data_loader_val, base_ds, device, args)
            ap50 = test_stats['coco_eval_bbox'][1]
            if ap50 > ap50_max:
                save_flag = True
                ap50_max = ap50
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if save_flag:
                checkpoint_paths = [output_dir / 'checkpoint_best.pth']
                save_flag = False
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 2 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if test_stats is not None:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth', f'{epoch:03}.pth']
                    # if epoch % 5 == 0:
                    #     filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

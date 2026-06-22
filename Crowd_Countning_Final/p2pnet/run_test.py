import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')


def compute_nap_single(pred_pts, pred_scores, gt_pts, delta, k=3):
    """
    Single-image nAP at a given delta threshold.
    Follows P2PNet paper (Song et al., ICCV 2021).
    Returns AP as a percentage (0~100).
    """
    if len(gt_pts) == 0:
        return 0.0

    gt_arr = np.array(gt_pts)

    # d_kNN for each GT point
    d_knn = np.ones(len(gt_arr))
    for i in range(len(gt_arr)):
        if len(gt_arr) <= 1:
            d_knn[i] = 1.0
            continue
        dists = np.sqrt(((gt_arr - gt_arr[i]) ** 2).sum(axis=1))
        dists[i] = np.inf
        k_use = min(k, len(gt_arr) - 1)
        d_knn[i] = np.sort(dists)[:k_use].mean()

    if len(pred_pts) == 0:
        return 0.0

    pred_arr = np.array(pred_pts)
    scores = np.array(pred_scores)
    order = np.argsort(-scores)
    pred_arr = pred_arr[order]
    scores = scores[order]

    matched = set()
    entries = []
    for pj, sj in zip(pred_arr, scores):
        best_i, best_ratio = -1, np.inf
        for i in range(len(gt_arr)):
            if i in matched:
                continue
            dist = np.sqrt(((pj - gt_arr[i]) ** 2).sum())
            ratio = dist / max(d_knn[i], 1e-6)
            if ratio < best_ratio:
                best_ratio = ratio
                best_i = i

        is_tp = int(best_i >= 0 and best_ratio < delta)
        if is_tp:
            matched.add(best_i)
        entries.append((float(sj), is_tp))

    entries.sort(key=lambda x: -x[0])
    tp_arr = np.array([e[1] for e in entries])
    cum_tp = np.cumsum(tp_arr)
    precision = cum_tp / (np.arange(len(tp_arr)) + 1)
    recall = cum_tp / len(gt_arr)

    ap = float(np.sum(precision * np.diff(np.concatenate([[0.0], recall]))))
    return ap * 100.0


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')
    parser.add_argument('--gt_path', default='',
                        help='(optional) path to GT annotation .txt file for nAP computation')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    img_path = "./vis/demo1.jpg"
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    threshold = 0.65
    # filter the predictions
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

    print(f'\nPredicted count: {predict_cnt}')

    # nAP computation (only when GT annotation file is provided)
    if args.gt_path and os.path.isfile(args.gt_path):
        gt_pts = []
        with open(args.gt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    gt_pts.append([float(parts[0]), float(parts[1])])

        pred_pts_all   = outputs_points[outputs_scores > threshold].detach().cpu().numpy()
        pred_scores_all = outputs_scores[outputs_scores > threshold].detach().cpu().numpy()

        print(f'GT count       : {len(gt_pts)}')
        print(f'MAE (this img) : {abs(predict_cnt - len(gt_pts))}')

        nap_005 = compute_nap_single(pred_pts_all, pred_scores_all, gt_pts, 0.05)
        nap_025 = compute_nap_single(pred_pts_all, pred_scores_all, gt_pts, 0.25)
        nap_050 = compute_nap_single(pred_pts_all, pred_scores_all, gt_pts, 0.50)

        print('\n[Localization – nAP (single image)]')
        print(f'  nAP  δ=0.05 : {nap_005:.1f}%')
        print(f'  nAP  δ=0.25 : {nap_025:.1f}%')
        print(f'  nAP  δ=0.50 : {nap_050:.1f}%')
    else:
        if args.gt_path:
            print(f'[경고] GT 파일을 찾을 수 없습니다: {args.gt_path}')
        print('[nAP 생략] --gt_path 를 지정하면 nAP를 계산합니다.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

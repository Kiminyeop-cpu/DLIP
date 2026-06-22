import argparse
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np
from torch.utils.data import DataLoader

from crowd_datasets.SHHA.SHHA import SHHA
from models import build_model
import util.misc as utils
import os
import warnings

warnings.filterwarnings('ignore')

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PART_B_TEST = str(_REPO_ROOT / 'data' / 'shanghaitech_p2p' / 'partB_test.list')
_PART_B_CKPT = str(_REPO_ROOT / 'p2pnet' / 'ckpt_partB' / 'partB_best_mae.pth')


def precompute_matching(all_pred_pts, all_pred_scores, all_gt_pts, k=3):
    """
    One-to-one greedy matching sorted by confidence (done once).
    Returns global list of (score, norm_ratio) and total_gt.
    norm_ratio=inf means unmatched (FP).
    """
    total_gt = sum(len(g) for g in all_gt_pts)
    global_entries = []

    for pred_pts, pred_sc, gt_pts in zip(all_pred_pts, all_pred_scores, all_gt_pts):
        if len(gt_pts) == 0:
            for s in pred_sc:
                global_entries.append((float(s), np.inf))
            continue

        gt_arr = np.array(gt_pts)

        # d_kNN – vectorized (N, N) pairwise distances
        if len(gt_arr) == 1:
            d_knn = np.array([1.0])
        else:
            diff = gt_arr[:, None, :] - gt_arr[None, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            k_use = min(k, len(gt_arr) - 1)
            d_knn = np.sort(dists, axis=1)[:, :k_use].mean(axis=1)

        if len(pred_pts) == 0:
            continue

        pred_arr = np.array(pred_pts)
        scores = np.array(pred_sc)
        order = np.argsort(-scores)
        pred_arr = pred_arr[order]
        scores = scores[order]

        # normalized distance matrix (M, N) – vectorized
        diff = pred_arr[:, None, :] - gt_arr[None, :, :]
        norm_mat = np.sqrt((diff ** 2).sum(axis=2)) / np.maximum(d_knn[None, :], 1e-6)

        matched = np.zeros(len(gt_arr), dtype=bool)
        for m_idx in range(len(pred_arr)):
            row = norm_mat[m_idx].copy()
            row[matched] = np.inf
            best_i = int(np.argmin(row))
            best_ratio = float(row[best_i])

            if best_ratio < np.inf:
                matched[best_i] = True
                global_entries.append((float(scores[m_idx]), best_ratio))
            else:
                global_entries.append((float(scores[m_idx]), np.inf))

    return global_entries, total_gt


def compute_nap(global_entries, total_gt, delta):
    """Compute nAP at given delta from pre-computed matching results."""
    if not global_entries or total_gt == 0:
        return 0.0

    sorted_entries = sorted(global_entries, key=lambda x: -x[0])
    tp_arr = np.array([1 if e[1] < delta else 0 for e in sorted_entries])
    cum_tp = np.cumsum(tp_arr)
    precision = cum_tp / (np.arange(len(tp_arr)) + 1)
    recall = cum_tp / total_gt

    ap = float(np.sum(precision * np.diff(np.concatenate([[0.0], recall]))))
    return ap * 100.0


def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet test – ShanghaiTech Part B', add_help=False)
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row',  default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--weight_path', default=_PART_B_CKPT)
    parser.add_argument('--val_list', default=_PART_B_TEST)
    parser.add_argument('--data_root', default='./new_public_density_data')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')

    model = build_model(args)
    model.to(device)

    checkpoint = torch.load(args.weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded weights : {args.weight_path}')
    print(f'Checkpoint epoch: {checkpoint.get("epoch", "unknown")}')
    model.eval()

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])

    val_set = SHHA(args.data_root, train=False, transform=transform,
                   eval_list=args.val_list)
    data_loader_val = DataLoader(val_set, 1,
                                 sampler=torch.utils.data.SequentialSampler(val_set),
                                 drop_last=False,
                                 collate_fn=utils.collate_fn_crowd,
                                 num_workers=args.num_workers)

    print(f'Test images: {len(val_set)}')
    print('Running evaluation on ShanghaiTech Part B ...\n')

    maes, mses, rel_errs = [], [], []
    all_pred_pts, all_pred_scores, all_gt_pts = [], [], []

    threshold = 0.65

    with torch.no_grad():
        for samples, targets in data_loader_val:
            samples = samples.to(device)
            outputs = model(samples)

            scores = torch.nn.functional.softmax(
                outputs['pred_logits'], -1)[:, :, 1][0]
            points = outputs['pred_points'][0]

            gt_pts = targets[0]['point'].numpy()
            gt_cnt = len(gt_pts)

            mask = scores > threshold
            pred_pts_np = points[mask].detach().cpu().numpy()
            pred_sc_np  = scores[mask].detach().cpu().numpy()
            pred_cnt = int(mask.sum())

            maes.append(abs(pred_cnt - gt_cnt))
            mses.append((pred_cnt - gt_cnt) ** 2)
            if gt_cnt > 0:
                rel_errs.append(abs(pred_cnt - gt_cnt) / gt_cnt * 100.0)

            all_pred_pts.append(pred_pts_np)
            all_pred_scores.append(pred_sc_np)
            all_gt_pts.append(gt_pts)

    mae     = np.mean(maes)
    mse     = np.sqrt(np.mean(mses))
    rel_pct = np.mean(rel_errs) if rel_errs else 0.0

    # matching once, reuse for all deltas
    global_entries, total_gt = precompute_matching(all_pred_pts, all_pred_scores, all_gt_pts)

    nap_005 = compute_nap(global_entries, total_gt, 0.05)
    nap_025 = compute_nap(global_entries, total_gt, 0.25)
    nap_050 = compute_nap(global_entries, total_gt, 0.50)

    print('=' * 50)
    print('  [Counting]')
    print(f'  MAE      : {mae:.4f}')
    print(f'  MSE      : {mse:.4f}')
    print(f'  rel_MAE% : {rel_pct:.2f}%')
    print()
    print('  [Localization – nAP]')
    print(f'  nAP  δ=0.05 : {nap_005:.1f}%')
    print(f'  nAP  δ=0.25 : {nap_025:.1f}%')
    print(f'  nAP  δ=0.50 : {nap_050:.1f}%')
    print('=' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet Part B test', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

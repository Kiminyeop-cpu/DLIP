"""
구간별 상대오차 평가 스크립트
- 각 이미지마다 rel_err = |pred - gt| / gt * 100% 를 개별 계산
- 구간(<50 / 50-100 / 100-300 / 300-600 / 600+)별로 따로 집계
- 구간 간 평균 절대 없음: 다른 밀도 구간을 섞으면 지표가 왜곡됨
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'p2pnet'))
from models import build_model

# ── 구간 정의 ────────────────────────────────────────────────────────────────
BINS = [
    (0,   50,  '<50',      '#4e79a7'),
    (50,  100, '50-100',   '#f28e2b'),
    (100, 300, '100-300',  '#e15759'),
    (300, 600, '300-600',  '#76b7b2'),
    (600, 10**9, '600+',  '#59a14f'),
]

def get_bin(gt):
    for lo, hi, label, color in BINS:
        if lo <= gt < hi:
            return label, color
    return '600+', BINS[-1][3]


def load_model(weight_path, device, row=2, line=2, backbone='vgg16_bn'):
    class Args:
        pass
    args = Args()
    args.backbone = backbone
    args.row = row
    args.line = line
    args.frozen_weights = None

    model = build_model(args, training=False)
    ckpt = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model


def infer_image(model, img_path, device, transform, threshold=0.65):
    img_raw = Image.open(img_path).convert('RGB')
    w, h = img_raw.size
    # P2PNet은 128 배수 크기를 요구함
    new_w = w // 128 * 128
    new_h = h // 128 * 128
    img_raw = img_raw.resize((new_w, new_h), Image.BILINEAR)
    img_t = transform(img_raw).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)

    scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[0, :, 1]
    pred_cnt = int((scores > threshold).sum())
    return pred_cnt


def read_gt(ann_path):
    points = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                points.append((float(parts[0]), float(parts[1])))
    return len(points)


def evaluate(list_path, weight_path, out_dir, split_name,
             threshold=0.65, backbone='vgg16_bn', row=2, line=2):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[{split_name}] device={device}, weights={os.path.basename(weight_path)}')

    model = load_model(weight_path, device, row=row, line=line, backbone=backbone)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # list 파일 읽기
    pairs = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            pairs.append((parts[0], parts[1]))

    print(f'[{split_name}] {len(pairs)} 이미지 평가 시작...')

    records = []  # (gt, pred, rel_err)
    for i, (img_path, ann_path) in enumerate(pairs):
        gt = read_gt(ann_path)
        pred = infer_image(model, img_path, device, transform, threshold)

        # gt=0인 경우: 예측도 0이면 완벽(0%), 아니면 100%로 처리
        if gt == 0:
            rel_err = 0.0 if pred == 0 else 100.0
        else:
            rel_err = abs(pred - gt) / gt * 100.0

        records.append((gt, pred, rel_err))

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(pairs)} 완료')

    records = sorted(records, key=lambda x: x[0])  # gt 기준 정렬

    # ── 구간별 집계 ──────────────────────────────────────────────────────────
    bin_data = {b[2]: [] for b in BINS}
    for gt, pred, rel_err in records:
        label, _ = get_bin(gt)
        bin_data[label].append((gt, pred, rel_err))

    print(f'\n{"=" * 68}')
    print(f'  [{split_name}] 구간별 평가 결과')
    print(f'{"=" * 68}')
    header = f'{"구간":>10}  {"n":>5}  {"평균 rel%":>10}  {"중앙 rel%":>10}  {"≤20% 비율":>10}'
    print(header)
    print('-' * 68)

    all_results = {}
    for lo, hi, label, color in BINS:
        rows = bin_data[label]
        n = len(rows)
        if n == 0:
            print(f'{label:>10}  {"0":>5}  {"N/A":>10}  {"N/A":>10}  {"N/A":>10}')
            all_results[label] = None
            continue
        errs = [r[2] for r in rows]
        mean_err  = np.mean(errs)
        med_err   = np.median(errs)
        acc20     = np.mean([e <= 20.0 for e in errs]) * 100.0
        print(f'{label:>10}  {n:>5}  {mean_err:>9.1f}%  {med_err:>9.1f}%  {acc20:>9.1f}%')
        all_results[label] = dict(n=n, mean_rel=mean_err, median_rel=med_err, acc20=acc20, rows=rows)

    print('=' * 68)
    print('  (bin-level only: no cross-bin averaging)')

    # ── 산점도 ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'P2PNet — {split_name}', fontsize=13)

    ax = axes[0]
    legend_patches = []
    for lo, hi, label, color in BINS:
        rows = bin_data[label]
        if not rows:
            continue
        gts   = [r[0] for r in rows]
        preds = [r[1] for r in rows]
        ax.scatter(gts, preds, c=color, alpha=0.6, s=18, label=label)
        legend_patches.append(mpatches.Patch(color=color, label=label))

    max_val = max(r[0] for r in records) if records else 100
    ax.plot([0, max_val], [0, max_val], 'k--', lw=1, alpha=0.5, label='perfect')
    ax.set_xlabel('GT count')
    ax.set_ylabel('Predicted count')
    ax.set_title('Pred vs GT (scatter)')
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color='k', lw=1, ls='--', label='perfect')],
              fontsize=8)

    # ── 구간별 상대오차 박스플롯 ─────────────────────────────────────────────
    ax2 = axes[1]
    plot_labels = []
    plot_data   = []
    plot_colors = []
    for lo, hi, label, color in BINS:
        rows = bin_data[label]
        if not rows:
            continue
        errs = [r[2] for r in rows]
        plot_labels.append(f'{label}\n(n={len(rows)})')
        plot_data.append(errs)
        plot_colors.append(color)

    bp = ax2.boxplot(plot_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(20, color='red', lw=1.2, ls='--', label='20% threshold')
    ax2.set_xticks(range(1, len(plot_labels) + 1))
    ax2.set_xticklabels(plot_labels, fontsize=8)
    ax2.set_ylabel('Relative error (%)')
    ax2.set_title('Relative error by bin')
    ax2.set_ylim(0, 100)  # 극단적 아웃라이어로 인한 축 압축 방지 (시각화 범위만 제한, 데이터는 그대로)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f'{split_name}_eval.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f'\n  [saved] {plot_path}')

    # ── CSV 저장 ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, f'{split_name}_per_image.csv')
    with open(csv_path, 'w') as f:
        f.write('gt,pred,rel_err_pct,bin\n')
        for gt, pred, rel_err in records:
            label, _ = get_bin(gt)
            f.write(f'{gt},{pred},{rel_err:.2f},{label}\n')
    print(f'  [saved] {csv_path}\n')

    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weight',     required=True, help='체크포인트 .pth 경로')
    ap.add_argument('--partA_test', required=True, help='partA_test.list 경로')
    ap.add_argument('--partB_test', required=True, help='partB_test.list 경로')
    ap.add_argument('--out_dir',    default='outputs/eval', help='결과 저장 디렉토리')
    ap.add_argument('--threshold',  default=0.65, type=float)
    ap.add_argument('--backbone',   default='vgg16_bn')
    ap.add_argument('--row',        default=2, type=int)
    ap.add_argument('--line',       default=2, type=int)
    args = ap.parse_args()

    evaluate(args.partA_test, args.weight, args.out_dir, 'PartA',
             threshold=args.threshold, backbone=args.backbone,
             row=args.row, line=args.line)

    evaluate(args.partB_test, args.weight, args.out_dir, 'PartB',
             threshold=args.threshold, backbone=args.backbone,
             row=args.row, line=args.line)


if __name__ == '__main__':
    main()

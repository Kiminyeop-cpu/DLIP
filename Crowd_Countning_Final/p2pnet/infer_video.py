"""
Video crowd counting inference
Usage:
    python infer_video.py --video 19743475-hd_1080_1920_30fps.mp4 --weight outputs/combined/ckpt/best_mae.pth
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'p2pnet'))
from models import build_model

# ── 라벨 기준 (숫자 조정은 여기서) ──────────────────────────────
THRESHOLDS = {
    'CROWD':   300,   # 300명 이상
    'CAUTION': 150,   # 150~299명
    # 150 미만 → SAFE
}
COLORS = {
    'CROWD':   (0, 0, 255),    # 빨강 (BGR)
    'CAUTION': (0, 165, 255),  # 주황
    'SAFE':    (0, 200, 0),    # 초록
}

EMA_ALPHA = 0.6  # 0~1, 클수록 현재값 반영 빠름 (0.3=부드럽게, 0.5=빠르게)


def get_label(count):
    if count >= THRESHOLDS['CROWD']:
        return 'CROWD'
    elif count >= THRESHOLDS['CAUTION']:
        return 'CAUTION'
    else:
        return 'SAFE'


def load_model(weight_path, device):
    class Args:
        backbone = 'vgg16_bn'
        row = 2
        line = 2
        frozen_weights = None
    model = build_model(Args(), training=False)
    ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model


def infer_frame(model, frame_bgr, device, transform, threshold=0.65):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    w, h = img.size
    new_w = max(w // 128 * 128, 128)
    new_h = max(h // 128 * 128, 128)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inp)
    scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[0, :, 1]
    points = outputs['pred_points'][0][scores > threshold].detach().cpu().numpy()
    points[:, 0] = points[:, 0] * w / new_w
    points[:, 1] = points[:, 1] * h / new_h
    return int(len(points)), points


def draw_overlay(frame, raw_count, smooth_count, label, points, draw_points=True):
    color = COLORS[label]
    h, w = frame.shape[:2]

    if draw_points:
        for p in points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)

    bar_h = 170
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f'Count: {smooth_count}',
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    cv2.putText(frame, label,
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 3.2, color, 8)

    box_w, box_h = 420, 150
    cv2.rectangle(frame, (w - box_w - 10, 10), (w - 10, 10 + box_h), color, -1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 6)[0]
    text_x = w - 10 - box_w // 2 - text_size[0] // 2
    text_y = 10 + box_h // 2 + text_size[1] // 2
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 6)

    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',     required=True)
    ap.add_argument('--weight',    required=True)
    ap.add_argument('--out',       default=None)
    ap.add_argument('--skip',      default=3, type=int, help='N프레임마다 추론')
    ap.add_argument('--threshold', default=0.65, type=float)
    ap.add_argument('--no_points', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = load_model(args.weight, device)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap    = cv2.VideoCapture(args.video)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video: {width}x{height} @ {fps:.1f}fps, {total} frames')

    out_path = args.out or os.path.splitext(args.video)[0] + '_result.mp4'
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    ema = None
    last_count  = 0
    last_points = np.zeros((0, 2))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.skip == 0:
            last_count, last_points = infer_frame(model, frame, device, transform, args.threshold)
            ema = last_count if ema is None else EMA_ALPHA * last_count + (1 - EMA_ALPHA) * ema

        smooth = int(round(ema)) if ema is not None else last_count
        label  = get_label(smooth)

        frame = draw_overlay(frame, last_count, smooth, label,
                             last_points, draw_points=not args.no_points)
        writer.write(frame)

        if frame_idx % (int(fps) * 5) == 0:
            print(f'  frame {frame_idx}/{total}  count={last_count}  smooth={smooth}  [{label}]')

        frame_idx += 1

    cap.release()
    writer.release()
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()

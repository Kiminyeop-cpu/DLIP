#!/usr/bin/env python3
"""
ShanghaiTech (.mat) -> P2PNet 점-리스트 포맷 변환기

각 이미지마다 머리 좌표 "x y"를 한 줄씩 담은 .txt(주석)를 만들고,
이미지<->주석을 매핑하는 .list 파일을 생성한다.

생성물:
  combined_train.list   (Part A train + Part B train)  <- 학습용(희소~밀집 전 구간)
  partA_test.list                                       <- 목표1 평가(밀집)
  partB_test.list                                       <- 목표1 평가(희소)
  partA_train.list / partB_train.list                   <- 따로 학습 비교 실험용

왜 통합 학습:
  모델 하나가 희소~밀집을 다 보게 해서, 배포 때 A/B 모델을 수동으로
  갈아끼우는(= 또 다른 오류원) 일을 없애기 위함.

사용:
  python prepare_shanghaitech.py \
      --data_root /path/to/ShanghaiTech \
      --out_root  /path/to/output
  (--data_root 안에 part_A, part_B 가 있어야 함)
"""

import argparse
import os
from glob import glob

import numpy as np
from scipy.io import loadmat


def load_points(mat_path):
    """ShanghaiTech GT .mat -> (N, 2) 머리 좌표(x, y) 배열."""
    mat = loadmat(mat_path)
    # 표준 ShanghaiTech 구조: image_info[0,0][0,0][0] == points (N x 2)
    points = mat["image_info"][0, 0][0, 0][0]
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def image_size(img_path):
    """(width, height) 반환. PIL 없으면 (None, None)."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size
    except Exception:
        return (None, None)


def clip_points(points, width, height):
    """이미지 경계 밖으로 살짝 벗어난 주석 제거(가끔 존재)."""
    if width is None or height is None:
        return points
    m = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    return points[m]


def convert_split(images_dir, gt_dir, ann_out_dir, do_clip):
    """한 split(train_data/test_data) 변환. (image_path, ann_path) 목록 반환."""
    os.makedirs(ann_out_dir, exist_ok=True)
    pairs = []
    img_paths = sorted(glob(os.path.join(images_dir, "IMG_*.jpg")))
    if not img_paths:
        print(f"  [경고] 이미지 없음: {images_dir}")
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]  # IMG_123
        gt_path = os.path.join(gt_dir, f"GT_{stem}.mat")
        if not os.path.exists(gt_path):
            print(f"  [경고] GT 없음, 건너뜀: {gt_path}")
            continue
        points = load_points(gt_path)
        if do_clip:
            w, h = image_size(img_path)
            points = clip_points(points, w, h)
        ann_path = os.path.join(ann_out_dir, f"{stem}.txt")
        # 머리 0개여도 빈 파일을 만들어 둠(희소/무인 장면 대비)
        np.savetxt(ann_path, points, fmt="%.2f")
        pairs.append((os.path.abspath(img_path), os.path.abspath(ann_path)))
    return pairs


def write_list(list_path, pairs):
    with open(list_path, "w") as f:
        for img_path, ann_path in pairs:
            f.write(f"{img_path} {ann_path}\n")
    print(f"  -> {list_path} ({len(pairs)} images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="ShanghaiTech 루트 (안에 part_A, part_B 존재)")
    ap.add_argument("--out_root", required=True,
                    help="생성된 .txt / .list 저장 위치")
    ap.add_argument("--no_clip", action="store_true",
                    help="경계 밖 주석 제거를 하지 않음 (PIL 불필요)")
    args = ap.parse_args()

    do_clip = not args.no_clip
    os.makedirs(args.out_root, exist_ok=True)

    all_pairs = {}
    for part in ["part_A", "part_B"]:
        for split in ["train_data", "test_data"]:
            images_dir = os.path.join(args.data_root, part, split, "images")
            gt_dir = os.path.join(args.data_root, part, split, "ground-truth")
            ann_out = os.path.join(args.out_root, part, split, "annotations")
            if not os.path.isdir(images_dir):
                print(f"[건너뜀] 폴더 없음: {images_dir}")
                all_pairs[(part, split)] = []
                continue
            print(f"[변환] {part}/{split}")
            all_pairs[(part, split)] = convert_split(
                images_dir, gt_dir, ann_out, do_clip
            )

    # 목적별 list 파일
    write_list(os.path.join(args.out_root, "combined_train.list"),
               all_pairs[("part_A", "train_data")]
               + all_pairs[("part_B", "train_data")])
    write_list(os.path.join(args.out_root, "partA_test.list"),
               all_pairs[("part_A", "test_data")])
    write_list(os.path.join(args.out_root, "partB_test.list"),
               all_pairs[("part_B", "test_data")])
    write_list(os.path.join(args.out_root, "partA_train.list"),
               all_pairs[("part_A", "train_data")])
    write_list(os.path.join(args.out_root, "partB_train.list"),
               all_pairs[("part_B", "train_data")])

    print("\n[요약] 변환된 이미지 수")
    for (part, split), v in all_pairs.items():
        counts = [sum(1 for _ in open(a)) for _, a in v] if v else []
        if counts:
            print(f"  {part}/{split}: {len(v)} imgs | "
                  f"인원 min {min(counts)} / 평균 {sum(counts)//len(counts)} "
                  f"/ max {max(counts)}")
        else:
            print(f"  {part}/{split}: 0 imgs")


if __name__ == "__main__":
    main()

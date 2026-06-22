"""
학습 로그 실시간 모니터링
python watch_train.py              # 한 번 출력
python watch_train.py --follow     # 30초마다 자동 갱신
"""

import re
import os
import sys
import time
import argparse

LOG_PATH = os.path.join(os.path.dirname(__file__),
                        'outputs', 'combined', 'log', 'run_log.txt')

def parse_log(path):
    with open(path, 'r') as f:
        raw = f.read()

    # 총 epochs 자동 파싱: Namespace(... epochs=<n> ...)
    total_epochs = None
    m = re.search(r'epochs=(\d+)', raw)
    if m:
        total_epochs = int(m.group(1))

    # eval_freq 자동 파싱
    eval_freq = 5
    m = re.search(r'eval_freq=(\d+)', raw)
    if m:
        eval_freq = int(m.group(1))

    # epoch별 loss
    losses = {}
    for m in re.finditer(r'loss/loss@(\d+): ([\d.e+-]+)', raw):
        losses[int(m.group(1))] = float(m.group(2))

    # 에폭별 소요시간
    times = {}
    lrs   = {}
    for m in re.finditer(r'\[ep (\d+)\]\[lr ([\d.]+)\]\[([\d.]+)s\]', raw):
        ep = int(m.group(1))
        lrs[ep]   = float(m.group(2))
        times[ep] = float(m.group(3))

    # MAE 이력 (rel_mae_pct 있는 버전 우선, 없으면 구버전 파싱)
    evals = []
    for m in re.finditer(
        r'mae:([\d.]+), mse:([\d.]+), rel_mae_pct:([\d.]+), time:([\d.]+), best mae:([\d.]+)', raw):
        evals.append({
            'mae':         float(m.group(1)),
            'mse':         float(m.group(2)),
            'rel_mae_pct': float(m.group(3)),
            'eval_sec':    float(m.group(4)),
            'best_mae':    float(m.group(5)),
        })
    if not evals:  # 구버전 로그 호환
        for m in re.finditer(
            r'mae:([\d.]+), mse:([\d.]+), time:([\d.]+), best mae:([\d.]+)', raw):
            evals.append({
                'mae':         float(m.group(1)),
                'mse':         float(m.group(2)),
                'rel_mae_pct': None,
                'eval_sec':    float(m.group(3)),
                'best_mae':    float(m.group(4)),
            })

    return losses, times, lrs, evals, total_epochs, eval_freq


def print_status(losses, times, lrs, evals, total_epochs, eval_freq=5):
    done_epochs = max(losses.keys()) + 1 if losses else 0
    pct = done_epochs / total_epochs * 100

    # 최근 10 에폭 평균 소요시간으로 남은 시간 추정
    recent_times = [times[e] for e in sorted(times)[-10:]] if times else []
    avg_sec = sum(recent_times) / len(recent_times) if recent_times else 0
    remaining_sec = avg_sec * (total_epochs - done_epochs)

    hrs, rem = divmod(int(remaining_sec), 3600)
    mins = rem // 60

    print('=' * 60)
    print(f'  P2PNet Combined A+B Training Monitor  ')
    print(f'  {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)
    print(f'  Progress : epoch {done_epochs} / {total_epochs}  ({pct:.1f}%)')
    print(f'  Speed    : {avg_sec:.1f}s / epoch')
    print(f'  ETA      : {hrs}h {mins}m')

    if losses:
        recent_eps = sorted(losses)[-10:]
        print(f'\n  [Recent 10 epoch loss]')
        for ep in recent_eps:
            mark = '>>' if ep == recent_eps[-1] else '  '
            print(f'  {mark} ep {ep:4d} | loss {losses[ep]:.5f} | {times.get(ep, 0):.1f}s')

    if evals:
        has_rel = any(e['rel_mae_pct'] is not None for e in evals)
        if has_rel:
            print(f'\n  [Eval history - Part A test]')
            print(f'  {"eval#":>6}  {"epoch":>6}  {"rel MAE%":>9}  {"abs MAE":>8}  {"MSE":>8}  {"best%":>7}')
            print(f'  ' + '-' * 57)
        else:
            print(f'\n  [Eval history - Part A test MAE (absolute)]')
            print(f'  {"eval#":>6}  {"epoch":>6}  {"MAE":>8}  {"MSE":>8}  {"best":>8}')
            print(f'  ' + '-' * 46)

        best_rel_list = []
        for i, e in enumerate(evals):
            eval_ep = (i + 1) * eval_freq
            best_mark = '<-' if abs(e['best_mae'] - e['mae']) < 0.01 else '  '
            if has_rel and e['rel_mae_pct'] is not None:
                best_rel_list.append(e['rel_mae_pct'])
                best_rel = min(best_rel_list)
                print(f'  {i+1:>6}  {eval_ep:>6}  {e["rel_mae_pct"]:>8.1f}%  '
                      f'{e["mae"]:>8.2f}  {e["mse"]:>8.2f}  {best_rel:>6.1f}% {best_mark}')
            else:
                print(f'  {i+1:>6}  {eval_ep:>6}  {e["mae"]:>8.2f}  {e["mse"]:>8.2f}  '
                      f'{e["best_mae"]:>8.2f} {best_mark}')

        if has_rel and best_rel_list:
            print(f'\n  Current best rel MAE = {min(best_rel_list):.1f}%  '
                  f'(abs MAE = {min(e["mae"] for e in evals):.1f})')
        else:
            best = min(e['best_mae'] for e in evals)
            print(f'\n  Current best MAE = {best:.2f}')
        print(f'  (ref: SHTechA pretrained ~68.0 abs / ~15-16% rel on PartA)')

    ckpt_path = os.path.join(os.path.dirname(__file__),
                             'outputs', 'combined', 'ckpt', 'best_mae.pth')
    if os.path.exists(ckpt_path):
        mtime = os.path.getmtime(ckpt_path)
        print(f'\n  best_mae.pth last saved: {time.strftime("%H:%M:%S", time.localtime(mtime))}')
    print('=' * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=LOG_PATH)
    ap.add_argument('--total', default=2000, type=int)
    ap.add_argument('--follow', action='store_true', help='30초마다 자동 갱신')
    ap.add_argument('--interval', default=30, type=int)
    args = ap.parse_args()

    while True:
        if not os.path.exists(args.log):
            print(f'Log not found: {args.log}')
        else:
            losses, times, lrs, evals, log_total, log_eval_freq = parse_log(args.log)
            total = log_total if log_total else args.total
            print_status(losses, times, lrs, evals, total_epochs=total, eval_freq=log_eval_freq)

        if not args.follow:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()

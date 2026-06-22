import argparse
import datetime
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings('ignore')

# Absolute paths to ShanghaiTech Part A list files
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PART_A_TRAIN = str(_REPO_ROOT / 'data' / 'shanghaitech_p2p' / 'partA_train.list')
_PART_A_TEST  = str(_REPO_ROOT / 'data' / 'shanghaitech_p2p' / 'partA_test.list')


def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet training – ShanghaiTech Part A', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    parser.add_argument('--frozen_weights', type=str, default=None)

    # Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str)

    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_point', default=0.05, type=float)

    # Loss
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)

    # Dataset – Part A defaults
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./new_public_density_data')
    parser.add_argument('--train_list', default=_PART_A_TRAIN)
    parser.add_argument('--val_list',   default=_PART_A_TEST)

    # Output – separate dirs so Part A weights never overwrite Part B
    parser.add_argument('--output_dir',      default='./log_partA')
    parser.add_argument('--checkpoints_dir', default='./ckpt_partA')
    parser.add_argument('--tensorboard_dir', default='./runs_partA')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)

    # Early stopping
    parser.add_argument('--early_stop_patience', default=10, type=int)
    parser.add_argument('--min_delta', default=0.01, type=float)

    parser.add_argument('--gpu_id', default=0, type=int)

    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    with open(run_log_name, "a") as log_file:
        log_file.write('\n[ShanghaiTech Part A] Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))

    device = torch.device('cuda')

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args, training=True)
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    loading_data = build_dataset(args=args)
    train_set, val_set = loading_data(args.data_root)

    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val   = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)
    data_loader_val   = DataLoader(val_set, 1, sampler=sampler_val,
                                   drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training – ShanghaiTech Part A")
    start_time = time.time()

    mae = []
    mse = []
    best_mae = float('inf')
    no_improve_count = 0

    writer = SummaryWriter(args.tensorboard_dir)
    step = 0

    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()

        stat = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)

        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}\n".format(epoch, stat['loss']))
                log_file.write("loss/loss_ce@{}: {}\n".format(epoch, stat['loss_ce']))
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]\n' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()

        # save latest checkpoint every epoch
        torch.save({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, os.path.join(args.checkpoints_dir, 'latest.pth'))

        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(model, data_loader_val, device)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])
            rel_pct = result[2]

            print('=======================================test=======================================')
            print("mae:", result[0], "mse:", result[1], "rel_mae_pct:", rel_pct,
                  "time:", t2 - t1, "best mae:", np.min(mae))
            with open(run_log_name, "a") as log_file:
                log_file.write("mae:{}, mse:{}, rel_mae_pct:{}, time:{}, best mae:{}\n".format(
                    result[0], result[1], rel_pct, t2 - t1, np.min(mae)))
            print('=======================================test=======================================')

            if writer is not None:
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

            current_mae = result[0]

            if current_mae < best_mae - args.min_delta:
                best_mae = current_mae
                no_improve_count = 0

                # Best checkpoint saved as partA_best_mae.pth
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, os.path.join(args.checkpoints_dir, 'partA_best_mae.pth'))

                print(f'New best MAE: {best_mae:.3f}')
                with open(run_log_name, "a") as log_file:
                    log_file.write(f'\nNew best MAE: {best_mae:.3f}\n')

            else:
                no_improve_count += 1
                print(f'No improvement: {no_improve_count}/{args.early_stop_patience}')
                with open(run_log_name, "a") as log_file:
                    log_file.write(f'\nNo improvement: {no_improve_count}/{args.early_stop_patience}\n')

            if no_improve_count >= args.early_stop_patience:
                print(f'Early stopping at epoch {epoch}. Best MAE: {best_mae:.3f}')
                with open(run_log_name, "a") as log_file:
                    log_file.write(f'\nEarly stopping at epoch {epoch}. Best MAE: {best_mae:.3f}\n')
                break

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Training time {}'.format(total_time_str))

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet Part A training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

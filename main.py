import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from utils.data_utils import get_loader
from medical.trainer import Trainer, Validator
from monai.inferers import SlidingWindowInferer
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import numpy as np
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from medical.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses.dice import DiceLoss
from monai.losses import FocalLoss
from medical.model.nested_former import NestedFormer

parser = argparse.ArgumentParser(description='NestedFormer segmentation pipeline for BRATS Challenge')

# 기본적인 인자 예시 (필요시 추가 확장하세요)
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--fold', default=4, type=int, help='data fold')
parser.add_argument('--data_dir', default='/tmp/pycharm_project_673/TrainingData', type=str, help='dataset directory')
parser.add_argument('--json_list', default='/tmp/pycharm_project_673/brats2020_datajson.json', type=str, help='dataset json file')
parser.add_argument('--max_epochs', default=300, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--sw_batch_size', default=4, type=int, help='sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimizer name (adam, adamw, sgd)')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum (for sgd)')
parser.add_argument('--val_every', default=10, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='enable distributed training')
parser.add_argument('--world_size', default=1, type=int, help='world size for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization method')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--feature_size', default=24, type=int, help='feature size')
parser.add_argument('--in_channels', default=4, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--cache_dataset', action='store_true', help='cache dataset flag')
parser.add_argument('--a_min', default=-175.0, type=float)
parser.add_argument('--a_max', default=250.0, type=float)
parser.add_argument('--b_min', default=0.0, type=float)
parser.add_argument('--b_max', default=1.0, type=float)
parser.add_argument('--space_x', default=1.0, type=float)
parser.add_argument('--space_y', default=1.0, type=float)
parser.add_argument('--space_z', default=1.0, type=float)
parser.add_argument('--roi_x', default=128, type=int)
parser.add_argument('--roi_y', default=128, type=int)
parser.add_argument('--roi_z', default=128, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--dropout_path_rate', default=0.0, type=float)
parser.add_argument('--RandFlipd_prob', default=0.2, type=float)
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float)
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float)
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float)
parser.add_argument('--infer_overlap', default=0.5, type=float)
parser.add_argument('--lrschedule', default='warmup_cosine', type=str)
parser.add_argument('--warmup_epochs', default=50, type=int)
parser.add_argument('--resume_ckpt', action='store_true')

def post_pred_func(pred):
    if isinstance(pred, (tuple, list)):  # deep supervision: multiple outputs
        pred = pred[0]  # validator에서는 첫 번째만 평가에 사용
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    return pred


def main():
    args = parser.parse_args()

    import sys
    from datetime import datetime

    os.makedirs('./runs', exist_ok=True)
    args.logdir = './runs/' + args.logdir
    os.makedirs(args.logdir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(args.logdir, f'train_log_{timestamp}.txt')

    class Tee:
        def __init__(self, name, mode):
            self.file = open(name, mode, encoding='utf-8')
            self.stdout = sys.stdout
            sys.stdout = self
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
        def flush(self):
            self.file.flush()
            self.stdout.flush()

    Tee(log_file_path, 'w')

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method('spawn', force=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = get_loader(args)

    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    model = NestedFormer(model_num=args.in_channels,
                         out_channels=args.out_channels,
                         image_size=inf_size,
                         window_size=(4, 4, 4),
                         )

    window_infer = SlidingWindowInferer(roi_size=inf_size,
                                        sw_batch_size=args.sw_batch_size,
                                        overlap=args.infer_overlap)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.', '')] = v
        model.load_state_dict(new_state_dict, strict=False)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print(f"=> loaded checkpoint '{args.checkpoint}' (epoch {start_epoch}) (bestacc {best_acc})")

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == 'batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          output_device=args.gpu)

    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      weight_decay=args.reg_weight)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.max_epochs)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    dice_metric = DiceMetric(include_background=True,
                             reduction=MetricReduction.MEAN_BATCH,
                             get_not_nans=True)

    validator = Validator(args,
                          model,
                          val_loader,
                          class_list=("TC", "WT", "ET"),
                          metric_functions=[["dice", dice_metric]],
                          sliding_window_infer=window_infer,
                          post_label=None,
                          post_pred=post_pred_func)

    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    focal_loss = FocalLoss()

    try:
        from medical.losses import CombinedLoss
        combined_loss = CombinedLoss([dice_loss, focal_loss], weights=[1.0, 1.0])
    except ImportError:
        print("medical.losses.CombinedLoss not found. Using DiceLoss only.")
        combined_loss = dice_loss

    trainer = Trainer(args,
                      train_loader,
                      validator=validator,
                      loss_func=combined_loss)

    best_acc, best_avg_dice = trainer.train(model,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            start_epoch=start_epoch)
    # 모델 저장 경로
    save_path = os.path.join(args.logdir, f'final_model_best_{best_avg_dice:.4f}.pt')


    # DDP 또는 일반 모델 처리
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    # 모델 저장
    torch.save(model_to_save.state_dict(), save_path)
    print(f"✅ Best model saved to: {save_path}")

    return best_acc


if __name__ == '__main__':
    main()

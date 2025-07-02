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
from medical.model.nested_former import NestedFormer
from medical.losses import CombinedLoss
import os
from datetime import datetime
import sys
from networks.vit_seg_modeling import VisionTransformer, CONFIGS



parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline for BRATS Challenge')
parser.add_argument('--drop_path_rate', default=0.1, type=float,
                    help='stochastic depth rate for EoFormer')

parser.add_argument('--model_name', default="NestedFormer", help='the model will be trained')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--fold', default=4, type=int, help='data fold')
parser.add_argument('--pretrain_model_path', default='./model.pt', type=str, help='pretrained model name')
parser.add_argument('--load_pretrain', action="store_true", help='pretrained model name')
parser.add_argument('--data_dir', default='/tmp/pycharm_project_802/TrainingData', type=str, help='dataset directory')
parser.add_argument('--json_list', default='/tmp/pycharm_project_802/TrainingData/brats2020_datajson.json', type=str, help='dataset json file')
parser.add_argument('--max_epochs', default=300, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=4, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--val_every', default=10, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization name')
parser.add_argument('--workers', default=2, type=int, help='number of workers')
parser.add_argument('--feature_size', default=24, type=int, help='feature size')
parser.add_argument('--in_channels', default=4, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--cache_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')

def post_pred_func(pred):
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    pred = torch.softmax(pred, dim=1)
    return torch.argmax(pred, dim=1, keepdim=True).float()


# ------------------ Main Entry ------------------
def main():
    args = parser.parse_args()
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
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)



def main_worker(gpu, args):
    # 기존 분산 초기화 부분 동일

    train_loader, val_loader = get_loader(args)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    # TransUNet config 가져오기
    config_vit = CONFIGS["R50-ViT-B_16"]  # 또는 다른 config ("ViT-B_16" 등)
    config_vit.n_classes = args.out_channels
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(args.roi_x / 16), int(args.roi_y / 16))  # grid 사이즈 설정

    model = VisionTransformer(config_vit, img_size=(args.roi_x, args.roi_y), num_classes=args.out_channels).cuda(gpu)


    # ✅ 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total trainable parameters: {total_params:,}")

    if args.distributed:
        if args.norm_name == 'batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    optimizer = {
        'adam': lambda: torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight),
        'adamw': lambda: torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight),
        'sgd': lambda: torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight),
    }[args.optim_name]()

    scheduler = {
        'warmup_cosine': LinearWarmupCosineAnnealingLR(optimizer, args.warmup_epochs, args.max_epochs),
        'cosine_anneal': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    }.get(args.lrschedule, None)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0)
    else:
        start_epoch, best_acc = 0, 0

    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)

    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    window_infer = SlidingWindowInferer(roi_size=inf_size, sw_batch_size=args.sw_batch_size, overlap=args.infer_overlap)

    validator = Validator(
        args, model, val_loader,
        class_list=("TC", "WT", "ET"),
        metric_functions=[["dice", dice_metric]],
        sliding_window_infer=window_infer,
        post_label=None,
        post_pred=post_pred_func
    )

    trainer = Trainer(args, train_loader, validator=validator, loss_func=dice_loss)

    best_acc = trainer.train(model, optimizer=optimizer, scheduler=scheduler, start_epoch=start_epoch)
    return best_acc


if __name__ == '__main__':
    main()

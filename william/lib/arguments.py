import json
import argparse

import torch

available_cuda_num = torch.cuda.device_count()

optimizer_choices = [
    name
    for name in torch.optim.__dict__
    if not name.startswith("_")
    and name[0].isupper()
    and callable(torch.optim.__dict__[name])
]

lr_scheduler_choices = [
    name
    for name in torch.optim.lr_scheduler.__dict__
    if name[0].isupper()
    and not name.startswith("_")
    and callable(torch.optim.lr_scheduler.__dict__[name])
] + ['CustomScheduler']


def parse_args():
    parser = argparse.ArgumentParser("Fined-grained food dataset")
    parser.add_argument("save_dir", type=str)
    parser.add_argument("--train-dir", type=str, default="/data/DLCV/food_data/train/")
    parser.add_argument("--valid-dir", type=str, default="/data/DLCV/food_data/val/")
    parser.add_argument(
        "--label2name-file", type=str, default="/data/DLCV/food_data/label2name.txt"
    )
    parser.add_argument("--seed", type=int, default=1116)
    parser.add_argument(
        "--gpu-ids", type=int, choices=list(range(-1, available_cuda_num)), default=0
    )
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--oversampling-thr", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument(
        "--optimizer", type=str, default="Adam", choices=optimizer_choices
    )
    parser.add_argument(
        "--optimizer-settings",
        type=json.loads,
        default={"lr": 3e-4, "weight_decay": 5e-4},
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=None,
        choices=lr_scheduler_choices,
    )
    parser.add_argument("--scheduler-settings", type=json.loads, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(available_cuda_num)

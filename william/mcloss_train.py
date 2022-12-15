import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.models import resnet18, resnet50, resnet101
from torchvision.datasets import ImageFolder
from torchvision import transforms as transforms

import os
import time

from tqdm import tqdm

from lib.utils import set_seeds, get_device
from lib.arguments import parse_args
from lib.dataset import CustomImageFolder, BalancedFoodDataset
from lib.scheduler import CustomScheduler
from lib.mcloss import MCLoss
from lib.model import BaseModel_scratch, BaseModel, Linear, Dense

try:
    from torch.utils.tensorboard import SummaryWriter

    use_tensorboard = True
except:
    use_tensorboard = False

import torch
import torch.nn as nn
from torchvision import models


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


train_transforms = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if __name__ == "__main__":

    args = parse_args()

    _ = set_seeds(args.seed)
    device = get_device(args.gpu_ids)

    train_loader = DataLoader(
        BalancedFoodDataset(
            CustomImageFolder(args.train_dir, train_transforms), args.oversampling_thr
        ),
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        ImageFolder(args.valid_dir, val_transforms),
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = BaseModel("s101", True)
    metric = Dense(2048, 1000)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    mcloss = MCLoss(num_classes=1000, cnums=[2, 3], cgroups=[952, 48])

    lr = args.optimizer_settings.pop("lr")
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr, **args.optimizer_settings)
    optimizer_metric = getattr(torch.optim, args.optimizer)(metric.parameters(), 1e-2, **args.optimizer_settings)

    use_scheduelr = False
    if args.lr_scheduler:
        use_scheduelr = True
        # scheduler = CustomScheduler(optimizer, schedule=args.scheduler_settings)
        scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(
            optimizer_metric, **args.scheduler_settings
        )

    model.to(device)
    metric.to(device)

    if use_tensorboard:
        writer = SummaryWriter()
        writer.add_text("args", str(vars(args)))
        writer.add_text("train_transform", str(train_transforms))
        writer.add_text("val_transform", str(val_transforms))

    print(str(vars(args)))

    print("*" * 35 + " Start training ! " + "*" * 35)

    with tqdm(total=args.num_epochs, desc="epoch") as e:

        best_acc = 0.0
        for epoch in range(args.num_epochs):
            model.train()
            metric.train()

            tacc = AverageMeter()
            tloss = AverageMeter()

            with tqdm(total=len(train_loader), desc="train iter.", leave=False) as t:
                for i, batch in enumerate(train_loader):

                    x, y = batch[0].to(device), batch[1].to(device)
                    feats, logits = model(x)
                    logits = metric(logits)

                    optimizer.zero_grad()
                    optimizer_metric.zero_grad()
                    loss = criterion(logits, y) + 0.005 * mcloss(feats, y)
                    loss.backward()
                    optimizer.step()
                    optimizer_metric.step()

                    acc = (torch.argmax(logits, dim=1) == y).float().mean().item()

                    tacc.update(acc)
                    tloss.update(loss.item())

                    if use_tensorboard:
                        writer.add_scalar(
                            "train/iter/acc", tacc.val, i + epoch * len(train_loader)
                        )
                        writer.add_scalar(
                            "train/iter/loss", tloss.val, i + epoch * len(train_loader)
                        )
                        writer.add_scalar(
                            "lr",
                            optimizer.param_groups[0]["lr"],
                            i + epoch * len(train_loader),
                        )

                    t.set_postfix(
                        {
                            "acc": tacc.val,
                            "loss": tloss.val,
                            "lr": optimizer.param_groups[0]["lr"],
                            "lr_metric": optimizer_metric.param_groups[0]["lr"]
                        }
                    )
                    t.update()
                    time.sleep(0.01)

                if use_scheduelr:
                    scheduler.step()

            vacc = AverageMeter()
            vloss = AverageMeter()

            model.eval()
            metric.eval()
            with tqdm(total=len(val_loader), desc="val iter.", leave=False) as t:
                for i, batch in enumerate(val_loader):

                    x, y = batch[0].to(device), batch[1].to(device)
                    
                    with torch.no_grad():
                        _, logits = model(x)
                        logits = metric(logits)
                        loss = criterion(logits, y)
                        acc = (torch.argmax(logits, dim=1) == y).float().mean().item()

                    vacc.update(acc)
                    vloss.update(loss.item())

                    if use_tensorboard:
                        writer.add_scalar(
                            "val/iter/acc", vacc.val, i + epoch * len(val_loader)
                        )
                        writer.add_scalar(
                            "val/iter/loss", vloss.val, i + epoch * len(val_loader)
                        )

                    t.set_postfix({"vacc": vacc.val, "vloss": vloss.val})
                    t.update()
                    time.sleep(0.01)

            if vacc.avg > best_acc:
                best_acc = vacc.avg
                e.write(f"Save best model @ epoch {epoch} @ acc {vacc.avg:.5f}!")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f"{save_dir}/best.pt",
                )

            if epoch % args.save_freq == 0:
                e.write(f"Save model @ epoch {epoch}!")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f"{save_dir}/{epoch}.pt",
                )

            if use_tensorboard:
                writer.add_scalar("train/epoch/acc", tacc.avg, epoch)
                writer.add_scalar("train/epoch/loss", tloss.avg, epoch)
                writer.add_scalar("val/epoch/acc", vacc.avg, epoch)
                writer.add_scalar("val/epoch/loss", vloss.avg, epoch)

            e.set_postfix(
                {
                    "tacc": tacc.avg,
                    "tloss": tloss.avg,
                    "vacc": vacc.avg,
                    "vloss": vloss.avg,
                }
            )
            e.update()

    print("*" * 35 + " Finish training ! " + "*" * 35)

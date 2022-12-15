import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets

from torchvision.models import resnet18, resnet50, resnet101
from torchvision.datasets import ImageFolder
from torchvision import transforms as transforms

import os
import time
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

from lib.utils import set_seeds, get_device
from lib.arguments import parse_args
from lib.dataset import CustomImageFolder, BalancedFoodDataset, PseudoSet
from lib.scheduler import CustomScheduler

try:
    from torch.utils.tensorboard import SummaryWriter

    use_tensorboard = True
except:
    use_tensorboard = False


def get_pseudo_labels(dataset, dataset_2, model, device, batch_size, threshold, transform=None):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    softmax = nn.Softmax(dim=-1)

    select_indices = []
    p_labels = []
    batch_id = 0
    for batch in tqdm(
        data_loader, desc="Making pseudo label", total=len(data_loader), leave=False
    ):
        img, _ = batch

        with torch.no_grad():
            logits = model(img.to(device))

        probs = softmax(logits)
        row, p_label = torch.where(probs > threshold)
        select_indices += (row + batch_id * batch_size).tolist()
        p_labels += p_label.tolist()
        assert len(select_indices) == len(p_labels)
        batch_id += 1

    pseudo_set = PseudoSet(dataset_2, select_indices, p_labels, transform)
    model.train()
    return pseudo_set


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
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomEqualize(0.1),
        transforms.RandomPosterize(2, 0.1),
        transforms.RandomSolarize(192, 0.1),
        transforms.RandomAdjustSharpness(2, 0.1),
        transforms.RandomGrayscale(0.1),
        transforms.RandomResizedCrop((224, 224), scale=(0.75, 0.95)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

pseudo_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomEqualize(0.1),
        transforms.RandomPosterize(2, 0.1),
        transforms.RandomSolarize(192, 0.1),
        transforms.RandomAdjustSharpness(2, 0.1),
        transforms.RandomGrayscale(0.1),
        transforms.RandomResizedCrop((224, 224), scale=(0.75, 0.95)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if __name__ == "__main__":

    args = parse_args()

    _ = set_seeds(args.seed)
    device = get_device(args.gpu_ids)

    train_set = BalancedFoodDataset(
        CustomImageFolder(args.train_dir, train_transforms), args.oversampling_thr
    )

    print(f'Num of training set: {len(train_set)}')
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    val_set = ImageFolder(args.valid_dir, val_transforms)
    val_loader = DataLoader(
        val_set, args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    psu_set = ImageFolder(args.valid_dir, val_transforms)
    psu_set_2 = ImageFolder(args.valid_dir, None)

    torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)
    # load pretrained models, using ResNeSt-50 as an example
    model = torch.hub.load("zhanghang1989/ResNeSt", "resnest101", pretrained=False)

    state_dict = torch.load(
        "/home/ubuntu/final-project-challenge-3-so_ez_peasy/william/s101_SGD_reload_semi_2/best.pt",
        map_location=device,
    )

    model.load_state_dict(state_dict["model_state_dict"])

    # model = resnet101(pretrained=True)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    with open("class_weight.pkl", "rb") as f:
        weights = pickle.load(f)
    weights = torch.from_numpy(np.array(list(weights.values()))).float().to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(), **args.optimizer_settings
    )

    use_scheduelr = False
    if args.lr_scheduler:
        use_scheduelr = True
        scheduler = CustomScheduler(optimizer, schedule=args.scheduler_settings)
        # scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(
        #     optimizer, **args.scheduler_settings
        # )

    model.to(device)

    if use_tensorboard:
        writer = SummaryWriter()
        writer.add_text("args", str(vars(args)))
        writer.add_text("train_transform", str(train_transforms))
        writer.add_text("val_transform", str(val_transforms))

    print(str(vars(args)))

    print("*" * 35 + " Start training ! " + "*" * 35)

    with tqdm(total=args.num_epochs, desc="epoch") as e:

        best_acc = 0.0
        do_pseudo_next_epoch = True
        for epoch in range(args.num_epochs):
            model.train()

            tacc = AverageMeter()
            tloss = AverageMeter()

            if do_pseudo_next_epoch:
                # Obtain pseudo-labels for unlabeled data using trained model.
                pseudo_set = get_pseudo_labels(psu_set, psu_set_2, model, device, args.batch_size, threshold=0.99, transform=train_transforms)

                # Construct a new dataset and a data loader for training.
                # This is used in semi-supervised learning only.
                concat_dataset = ConcatDataset([train_set, pseudo_set])
                train_loader = DataLoader(
                    concat_dataset,
                    args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    drop_last=True
                )
                e.write(f"Num of pseudo image: {len(pseudo_set)}")
            else:
                train_loader = DataLoader(
                    train_set, args.batch_size, shuffle=True, num_workers=args.num_workers
                )
                e.write("No psuedo")

            with tqdm(total=len(train_loader), desc="train iter.", leave=False) as t:
                for i, batch in enumerate(train_loader):

                    x, y = batch[0].to(device), batch[1].to(device)
                    logits = model(x)

                    optimizer.zero_grad()
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

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
                        }
                    )
                    t.update()
                    time.sleep(0.01)

                if use_scheduelr:
                    scheduler.step()

            vacc = AverageMeter()
            vloss = AverageMeter()

            model.eval()
            with tqdm(total=len(val_loader), desc="val iter.", leave=False) as t:
                for i, batch in enumerate(val_loader):

                    x, y = batch[0].to(device), batch[1].to(device)
                    logits = model(x)

                    with torch.no_grad():
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

            if epoch + 1 == args.num_epochs:
                e.write(f"Save latest model !")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f"{save_dir}/latest.pt",
                )

            if vacc.avg > 0.7:
                do_pseudo_next_epoch = True
            else:
                do_pseudo_next_epoch = False

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

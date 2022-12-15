import torch
from torch.utils.data import DataLoader, Dataset, dataset
from torchvision.models import resnet101
from torchvision import transforms

import os
import sys
import csv
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from lib.utils import set_seeds, get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--test-dir", type=str, default="/home/ubuntu/data/DLCV/food_data/test")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/b05611046/DLCV/final-project-challenge-3-so_ez_peasy/william/s101_SGD_multistep_1e-5/best.pt",
    )
    parser.add_argument("--seed", type=int, default=1116)
    parser.add_argument(
        "--gpu-ids",
        type=int,
        choices=list(range(-1, torch.cuda.device_count())),
        default=0,
    )
    return parser.parse_args()


class FoodDataset(Dataset):
    def __init__(self, root, transform=None, names=None):
        super().__init__()

        self.root = root
        self.transform = transform
        self.images = [
            img for img in sorted(Path(root).glob("*.jpg")) if str(img.stem) in names
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.images[idx])), str(self.images[idx].stem)

def get_testcase_names(testcase_csv):
    with open(testcase_csv, "r") as case:
        reader = csv.reader(case)
        next(reader, None)
        case_dict = {str(row[0]): np.array(row[1]).astype(int) for row in reader}
    
    return case_dict.keys()

if __name__ == "__main__":
    args = parse_args()

    testcase_csv = lambda file: f"/home/ubuntu/data/DLCV/food_data/testcase/sample_submission_{file}_track.csv"
    testcase_ids = list(map(get_testcase_names, list(map(testcase_csv, ['main', 'freq', 'comm', 'rare']))))

    set_seeds(args.seed)
    device = get_device(args.gpu_ids)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    with open("cvt_dict.pkl", "rb") as f:
        cvt_dict = pickle.load(f)

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dst = FoodDataset(args.test_dir, val_transforms, testcase_ids[0])
    test_loader = DataLoader(test_dst, batch_size=32, shuffle=False, num_workers=10)

    torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)
    # load pretrained models, using ResNeSt-50 as an example
    model = torch.hub.load("zhanghang1989/ResNeSt", "resnest101", pretrained=False)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)
    model.eval()

    preds = []
    names = []
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='testing'):
        img, name = batch[0].to(device), batch[1]
        with torch.no_grad():
            pred = torch.argmax(model(img), dim=1).cpu().numpy()
            preds.append(pred)
            names.append(name)

    preds = np.concatenate(preds).flatten()
    cvt_preds = np.array([cvt_dict[p] for p in preds]).reshape(-1, 1).astype(int)
    names = np.concatenate(names).flatten().reshape(-1, 1)
    results = np.hstack((names, cvt_preds))

    # assert cvt_preds.shape[0] == len(case_dict), "length of prediction not match"

    maincsv = open(os.path.join(args.output_dir, 'main.csv'), 'w')
    freqcsv = open(os.path.join(args.output_dir, 'freq.csv'), 'w')
    commcsv = open(os.path.join(args.output_dir, 'comm.csv'), 'w')
    rarecsv = open(os.path.join(args.output_dir, 'rare.csv'), 'w')

    mainwriter = csv.writer(maincsv)
    freqwriter = csv.writer(freqcsv)
    commwriter = csv.writer(commcsv)
    rarewriter = csv.writer(rarecsv)

    mainwriter.writerow(np.array(["image_id", "label"]))
    freqwriter.writerow(np.array(["image_id", "label"]))
    commwriter.writerow(np.array(["image_id", "label"]))
    rarewriter.writerow(np.array(["image_id", "label"]))

    for row in results:
        mainwriter.writerow(row)

        if row[0] in testcase_ids[1]: #freq
            freqwriter.writerow(row)
        if row[0] in testcase_ids[2]: #comm
            commwriter.writerow(row)
        if row[0] in testcase_ids[3]: #rare
            rarewriter.writerow(row)

    maincsv.close()
    freqcsv.close()
    commcsv.close()
    rarecsv.close()

    print("Finish predicting !")

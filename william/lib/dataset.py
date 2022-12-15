import os
import math
import pickle
import numpy as np

from collections import defaultdict

from PIL import Image
from torchvision.datasets.folder import VisionDataset
from torch.utils.data import Dataset, Subset


def find_classes(directory: str):

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)} #folder_name: idx
    return classes, class_to_idx


def make_dataset(
    directory: str, class_to_idx=None, extensions=None, is_valid_file=None
):

    directory = os.path.expanduser(directory)

    instances = []
    available_classes = set()
    classes_fraction = {k: 0 for k in range(1000)}
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)

                item = path, class_index
                classes_fraction[class_index] += 1
                instances.append(item)

                if target_class not in available_classes:
                    available_classes.add(target_class)
    for id, _ in classes_fraction.items():
        classes_fraction[id] /= len(instances)
    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = (
            f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        )
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances, classes_fraction


def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class CustomImageFolder(VisionDataset):
    def __init__(
        self,
        root,
        transform=None,
        loader=pil_loader,
        extensions=None,
        target_transform=None,
        is_valid_file=None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples, self.classes_fraction = self.make_dataset(
            self.root, class_to_idx, extensions, is_valid_file
        )

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file
        )

    def find_classes(self, directory: str):

        return find_classes(directory)

    def __getitem__(self, index: int):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class BalancedFoodDataset:
    def __init__(self, dataset, oversample_thr):

        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.CLASSES = dataset.classes

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

    def _get_repeat_factors(self, dataset, repeat_thr):
        """Get repeat factor for each images in the dataset.
        Args:
            dataset (:obj:`CustomDataset`): The dataset
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.
        Returns:
            list[float]: The repeat factors for each images in the dataset.
        """

        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = self.dataset.classes_fraction
        num_images = len(dataset)
        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = list(range(1000))
            repeat_factor = 1
            if len(cat_ids) > 0:
                repeat_factor = max({category_repeat[cat_id] for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        """Length after repetition."""
        return len(self.repeat_indices)

class PseudoSet(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """

    def __init__(self, dataset, indices, labels, transform):
        self.dataset = Subset(dataset, indices)
        self.targets = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return self.transform(image), target

    def __len__(self):
        return len(self.targets)


if __name__ == "__main__":
    from torchvision import transforms as transforms

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224)),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    cif = CustomImageFolder("/home/ubuntu/data/DLCV/food_data/train/", train_transforms)
    # cvt_dict = {v:int(k) for k, v in cif.class_to_idx.items()}
    # print(cvt_dict)
    # with open('cvt_dict.pkl', 'wb') as f:
    #     pickle.dump(cvt_dict, f)
    
    bds = BalancedFoodDataset(cif, 0.00000715)
    print(len(bds))
    # a = 0
    # for k, v in cif.classes_fraction.items():
    #     if v <= 0.0001:
    #         a += 1
    # print(a)

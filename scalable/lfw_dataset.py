"""LFW dataloading."""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm

from pathlib import Path


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.savefig("batch_viz.png")
    plt.show()


class LFWDataset(Dataset):
    """Initialize LFW dataset."""

    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.transform = transform

        self.data = list(Path(path_to_folder).glob("**/*.jpg"))

    def __len__(self):
        """Return length of dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get item from dataset."""
        img = Image.open(self.data[index])

        return self.transform(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="data/lfw", type=str)
    parser.add_argument("-batch_size", default=16, type=int)
    parser.add_argument("-num_workers", default=0, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)
    parser.add_argument("-errorbar", action="store_true")

    args = parser.parse_args()

    lfw_trans = transforms.Compose(
        [transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)), transforms.ToTensor()]
    )

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    if args.visualize_batch:
        plt.rcParams["savefig.bbox"] = "tight"
        imgs = []
        batch = next(iter(dataloader))
        for e in batch:
            imgs.append(e.squeeze(0))

        grid = make_grid(imgs)
        show(grid)

    if args.get_timing:
        # lets do some repetitions
        res = []
        for _ in range(5):
            start = time.time()
            for batch_idx, _batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)

        res = np.array(res)
        mn = np.mean(res)
        std_d = np.std(res)
        print(f"Timing: {np.mean(res)}+-{np.std(res)}")

    if args.errorbar:
        # lets do some repetitions
        num_w = args.num_workers
        if num_w > 4:
            num_w = 4

        xs = []
        ys = []
        stds_ = []
        for i in tqdm(range(0, num_w + 1)):
            xs.append(i)
            res = []
            for _ in range(5 * 4):
                start = time.time()
                for batch_idx, _batch in enumerate(dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()

                res.append(end - start)

            res = np.array(res)
            mn = np.mean(res)
            std_d = np.std(res)
            ys.append(mn)
            stds_.append(std_d)

        plt.errorbar(xs, ys, yerr=stds_)
        plt.show()

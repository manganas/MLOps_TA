# -*- coding: utf-8 -*-
import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import click
import torch
import wget
from torch import Tensor
from torch.utils.data import Dataset

from torchvision import transforms


class CorruptMnist(Dataset):
    def __init__(self, train: bool, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()

        self.train = train
        self.in_folder = in_folder
        self.out_folder = out_folder

        return

        if self.train:
            content = []
            for i in range(8):
                content.append(
                    torch.load(f"{in_folder}/train_{i}.pt", allow_pickle=True)
                )
            data = torch.tensor(np.concatenate([c["images"] for c in content])).reshape(
                -1, 1, 28, 28
            )
            targets = torch.tensor(np.concatenate([c["labels"] for c in content]))
        else:
            content = np.load(f"{in_folder}/test.npz", allow_pickle=True)
            data = torch.tensor(content["images"]).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content["labels"])

        self.data = data
        self.targets = targets

        if self.out_folder:
            self.save_preprocessed()

    def save_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        torch.save([self.data, self.targets], f"{self.out_folder}/{split}_processed.pt")

    def load_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        try:
            self.data, self.targets = torch.load(
                f"{self.out_folder}/{split}_processed.pt"
            )
        except:
            raise ValueError("No preprocessed files found")

    
    def preprocess_data(self) -> None:
        if not Path(self.out_folder).exists():
            print("Creating preprocessed data folder...")
            Path(self.out_folder).mkdir(parents=True, exist_ok=True)

        
        transf = transforms.Compose(
            [
                transforms.Normalize((0.0,), (1.0,)),
            ]
        )

        if self.train:
            images = []
            targets = []

            for i in range(9):
                img = torch.load(f"{self.in_folder}/train_images_{i}.pt")
                images.append(
                    transf(img)
                )               

                targets.append(torch.load(f"{self.in_folder}/train_target_{i}.pt"))

            images = torch.concat(images, dim=0)

            targets = torch.concat(targets, dim=0)

            torch.save(images.unsqueeze(1), f"{self.out_folder}/train_images.pt")
            torch.save(targets, f"{self.out_folder}/train_targets.pt")

        else:

            test_images_file_path = Path(self.out_folder) / "test_images.pt"
            if test_images_file_path.exists():
                print(f"{test_images_file_path} already exists, skipping preprocessing")
            else:
                images = transf(torch.load(f"{self.in_folder}/test_images.pt"))
                torch.save(images.unsqueeze(1), f"{self.out_folder}/test_images.pt")
            
            
            test_targets_file_path = Path(self.out_folder) / "test_targets.pt"
            if test_targets_file_path.exists():
                print(f"{test_targets_file_path} already exists, skipping preprocessing")
            else:
                targets = torch.load(f"{self.in_folder}/test_target.pt")
                torch.save(targets, f"{self.out_folder}/test_targets.pt")





    def download_data(self) -> None:
        if not Path(self.in_folder).exists():
            print("Creating raw data folder...")
            Path(self.in_folder).mkdir(parents=True, exist_ok=True)

        files = os.listdir(self.in_folder)
        if self.train:
            for file_idx in range(10):
                if file_idx < 6:
                    if f"train_images_{file_idx}.pt" not in files:
                        wget.download(
                            f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_images_{file_idx}.pt"
                        )
                        shutil.move(
                            f"train_images_{file_idx}.pt",
                            f"{self.in_folder}/train_images_{file_idx}.pt",
                        )
                    if f"train_target_{file_idx}.pt" not in files:
                        wget.download(
                            f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/train_target_{file_idx}.pt"
                        )
                        shutil.move(
                            f"train_target_{file_idx}.pt",
                            f"{self.in_folder}/train_target_{file_idx}.pt",
                        )

                if file_idx > 5 and f"train_{file_idx}.pt" not in files:
                    if f"train_images_{file_idx}.pt" not in files:
                        wget.download(
                            f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist_v2/train_images_{file_idx}.pt"
                        )
                        shutil.move(
                            f"train_images_{file_idx}.pt",
                            f"{self.in_folder}/train_images_{file_idx}.pt",
                        )
                    if f"train_target_{file_idx}.pt" not in files:
                        wget.download(
                            f"https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist_v2/train_target_{file_idx}.pt"
                        )
                        shutil.move(
                            f"train_target_{file_idx}.pt",
                            f"{self.in_folder}/train_target_{file_idx}.pt",
                        )
        else:
            if "test_images.pt" not in files:
                wget.download(
                    "https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test_images.pt"
                )
                shutil.move("test_images.pt", f"{self.in_folder}/test_images.pt")

            if "test_target.pt" not in files:
                wget.download(
                    "https://raw.githubusercontent.com/SkafteNicki/dtu_mlops/main/data/corruptmnist/test_target.pt"
                )
                shutil.move("test_target.pt", f"{self.in_folder}/test_target.pt")

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx].float(), self.targets[idx]


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True), default="data/raw")
@click.argument("output_filepath", type=click.Path(), default="data/processed")
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train = CorruptMnist(
        train=True, in_folder=input_filepath, out_folder=output_filepath
    )
    train.download_data()
    print("Train data downloaded")
    train.preprocess_data()

    test = CorruptMnist(
        train=False, in_folder=input_filepath, out_folder=output_filepath
    )

    test.download_data()
    print("Test data downloaded")
    test.preprocess_data()

    # load saved files to print dimensions
    
    train_images_tensor = torch.load(f"{output_filepath}/train_images.pt")
    train_targets_tensor = torch.load(f"{output_filepath}/train_targets.pt")

    print(f"Train images shape: {train_images_tensor.shape}")
    print(f"Train targets shape: {train_targets_tensor.shape}")

    test_images_tensor = torch.load(f"{output_filepath}/test_images.pt")
    test_targets_tensor = torch.load(f"{output_filepath}/test_targets.pt")
    print(f"Test images shape: {test_images_tensor.shape}")
    print(f"Test targets shape: {test_targets_tensor.shape}")

    
    # print(train.data.shape)
    # print(train.targets.shape)
    # print(test.data.shape)
    # print(test.targets.shape)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

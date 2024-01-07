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
    def __init__(self, train: bool, in_folder: str = "", out_folder: str = "", transf=None) -> None:
        super().__init__()

        self.train = train
        self.in_folder = in_folder
        self.out_folder = out_folder

        self.transf = transf
        if not transf:
            self.transf = transforms.Compose(
                [
                    transforms.Normalize((0.0,), (1.0,)),
                ]
            )

        if self.train:
            try:  # Train dataset
                self.data = torch.load(f"{self.out_folder}/train_images.pt")
                self.targets = torch.load(f"{self.out_folder}/train_targets.pt")
            except FileNotFoundError:
                self.download_data()
                self.preprocess_data()
                self.data = torch.load(f"{self.out_folder}/train_images.pt")
                self.targets = torch.load(f"{self.out_folder}/train_targets.pt")
        else:  # Test dataset
            try:
                self.data = torch.load(f"{self.out_folder}/test_images.pt")
                self.targets = torch.load(f"{self.out_folder}/test_targets.pt")
            except FileNotFoundError:
                self.download_data()
                self.preprocess_data()
                self.data = torch.load(f"{self.out_folder}/test_images.pt")
                self.targets = torch.load(f"{self.out_folder}/test_targets.pt")

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx].float(), self.targets[idx]

    def __len__(self) -> int:
        return self.targets.numel()

    def preprocess_data(self) -> None:
        """
        Preprocesses the data and saves it to the out_folder
        """
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
                images.append(transf(img))

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
        """
        Downloads the data from github if it does not already exist in the raw data folder
        """
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


@click.command()
@click.argument("input_filepath", type=click.Path(), default="data/raw")
@click.argument("output_filepath", type=click.Path(), default="data/processed")
def main(input_filepath: str, output_filepath: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train = CorruptMnist(train=True, in_folder=input_filepath, out_folder=output_filepath)
    train.download_data()
    print("Train data downloaded")
    train.preprocess_data()
    print("Train data processed\n")

    test = CorruptMnist(train=False, in_folder=input_filepath, out_folder=output_filepath)

    test.download_data()
    print("Test data downloaded")
    test.preprocess_data()
    print("Test data processed")

    # load saved files to print dimensions

    print("\n\nTesting processed shapes:\n")

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

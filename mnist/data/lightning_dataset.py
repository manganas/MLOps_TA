import torch
from torch.utils.data import DataLoader, random_split

from pytorch_lightning import LightningDataModule

from mnist.data.make_dataset import CorruptMnist


class CorruptMnistLightningModule(LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 32, seed=42) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = CorruptMnist(
                train=True, out_folder=self.data_dir, in_folder="data/raw"
            )

            # split mnist_full dataset
            train_size = int(0.8 * len(mnist_full))
            test_size = len(mnist_full) - train_size
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(
                mnist_full, [train_size, test_size]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = CorruptMnist(
                train=False, out_folder=self.data_dir, in_folder="data/raw"
            )

        if stage == "predict":
            self.mnist_test = CorruptMnist(
                train=False, out_folder=self.data_dir, in_folder="data/raw"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
        )

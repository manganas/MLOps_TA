import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist.models.model import SimpleCNN
from mnist.data.make_dataset import CorruptMnist

from tqdm import tqdm
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra

import logging


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    ## Hparams
    seed = config.seed
    set_seed(seed)

    # paths
    data_folder = Path(config.data_path)
    in_folder = data_folder / "raw"
    out_folder = data_folder / "processed"

    save_model_path = config.model_path
    Path(save_model_path).mkdir(parents=True, exist_ok=True)

    # training params
    batch_size = config.experiment.batch_size
    epochs = config.experiment.num_epochs
    lr = config.experiment.lr
    img_size = config.experiment.img_size
    num_classes = config.experiment.num_classes

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Runs on: {device}")

    # Datasets
    train_dataset = CorruptMnist(train=True, in_folder=in_folder, out_folder=out_folder)

    # split training dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = SimpleCNN(img_size=img_size, out_features=num_classes)
    # maybe try to load a previously saved model
    model.to(device)

    # Loss and optimizer
    loss = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in pbar:
            optim.zero_grad()
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            y_hat = model(images)
            l = loss(y_hat, targets)
            l.backward()
            optim.step()

            train_loss += l.item()
            train_acc += (y_hat.argmax(1) == targets).sum().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        for batch in pbar:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                y_hat = model(x)
                l = loss(y_hat, y)

            val_loss += l.item()
            val_acc += (y_hat.argmax(1) == y).sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), save_model_path + "/model.pt")


if __name__ == "__main__":
    main()

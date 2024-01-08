import torch
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from mnist.models.model_L import SimpleCNN
from mnist.data.lightning_dataset import CorruptMnistLightningModule

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    img_size = cfg.experiment.img_size
    num_classes = cfg.experiment.num_classes
    n_epochs = cfg.experiment.num_epochs

    dm = CorruptMnistLightningModule("data/processed", batch_size=cfg.experiment.batch_size)

    model = SimpleCNN(img_size=img_size, out_features=num_classes)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="", mode="min")

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    callbacks = [checkpoint_callback, early_stopping_callback]

    trainer = Trainer(max_epochs=n_epochs, callbacks=callbacks)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader

from mnist.models.model import SimpleCNN
from mnist.data.make_dataset import CorruptMnist


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch, _ in dataloader], 0)


def main():
    # hparams
    batch_size = 32
    img_size = 28
    num_classes = 10

    test_path = "data/processed"

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Predixtion runs on: {device}")

    # dataset
    test_dataset = CorruptMnist(train=False, in_folder="", out_folder=test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(img_size=img_size, out_features=num_classes)
    try:
        model.load_state_dict(torch.load("models/model.pt"))
    except FileNotFoundError:
        print("No model found. Please train a model first.")
        # exit()

    model.to(device)

    predictions = predict(model, test_loader)

    for i, pred in enumerate(predictions):
        print(torch.argmax(pred).item(), test_dataset[i][1].item())


if __name__ == "__main__":
    main()

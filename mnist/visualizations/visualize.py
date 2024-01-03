import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from tqdm import tqdm
from pathlib import Path


from mnist.models.model import SimpleCNN
from mnist.data.make_dataset import CorruptMnist

# import os

# default_n_threads = 1
# os.environ["OPENBLAS_NUM_THREADS"] = f"{default_n_threads}"
# # os.environ["MKL_NUM_THREADS"] = f"{default_n_threads}"
# # os.environ["OMP_NUM_THREADS"] = f"{default_n_threads}"
# os.environ["OPENBLAS_VERBOSE"] = f"2"


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def tsne_vis():
    # haparams
    seed = 42
    set_seed(seed)

    batch_size = 32
    img_size = 28
    num_classes = 10

    n_components = 2

    save_path = Path("reports/figures")
    save_path.mkdir(exist_ok=True, parents=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset and loader
    train_set = CorruptMnist(train=True, in_folder="", out_folder="data/processed")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # model
    model = SimpleCNN(img_size=img_size, out_features=num_classes)
    try:
        model.load_state_dict(torch.load("models/model.pt"))
    except FileNotFoundError:
        print("No model found. Please train a model first.")
        exit()
    model.to(device)

    embeddings, labels = [], []

    with torch.no_grad():
        pbar = tqdm(train_loader)
        for batch in pbar:
            data, targets = batch
            data = data.to(device)

            emb = model.cnn_net(data)

            embeddings.append(emb.reshape(data.shape[0], -1))  # or flatten
            labels.append(targets)

    embeddings = torch.cat(embeddings, 0).cpu().numpy()
    labels = torch.cat(labels, 0).numpy()

    # embeddings and labels should be numpy arrays
    print("Running tSNE...")
    tsne = TSNE(n_components=n_components, random_state=seed, n_jobs=1)

    emb_2d = tsne.fit_transform(embeddings)

    print("Plotting...")

    for i in np.unique(labels):
        plt.scatter(emb_2d[labels == i, 0], emb_2d[labels == i, 1], label=f"{i}")

    plt.legend()
    save_name = save_path / "tsne.png"
    plt.savefig(save_name)


if __name__ == "__main__":
    tsne_vis()

import torch
import torch.nn as nn


class SimpleCNN(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, img_size: int, out_features: int) -> None:
        super().__init__()

        self.cnn_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out = self._get_cnn_out(img_size)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def _get_cnn_out(self, img_size: int) -> int:
        """Computes the output of the CNN part of the model.

        Args:
            img_size: size of the input image

        Returns:
            Output of the CNN part of the model

        """
        out = self.cnn_net(torch.zeros(1, 1, img_size, img_size))
        return out.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        if x.ndim != 4:
            raise ValueError("Expected input to be a 4D tensor")

        x = self.cnn_net(x)
        logits = self.classifier(x)
        if logits.shape[-1] != 10:
            raise ValueError(f"Expected output shape of [N,10]")
        return logits

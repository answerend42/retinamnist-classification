"""Timm-based classifier wrapper for easy model selection."""

import timm
import torch.nn as nn


class TimmClassifier(nn.Module):
    """Wrapper for timm models to use in Lightning-Hydra framework.

    Args:
        model_name: Name of the timm model (e.g., 'resnet18', 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        drop_rate: Dropout rate
        drop_path_rate: Drop path rate for stochastic depth
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 1000,
        pretrained: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    # Test the model
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test ResNet-18
    model = TimmClassifier(model_name="resnet18", num_classes=5).to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # List available models
    print("\nAvailable ResNet models:")
    resnet_models = timm.list_models("resnet*", pretrained=True)
    for m in resnet_models[:10]:
        print(f"  - {m}")

    print("\nAvailable EfficientNet models:")
    effnet_models = timm.list_models("efficientnet*", pretrained=True)
    for m in effnet_models[:10]:
        print(f"  - {m}")
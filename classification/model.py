import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU → MaxPool with residual shortcut."""
    def __init__(self, in_ch: int, out_ch: int, pool: tuple = (2, 2)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.pool  = nn.MaxPool2d(pool)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + F.adaptive_avg_pool2d(residual, x.shape[2:])
        return self.pool(x)


class BirdsongCNN(nn.Module):
    """
    Lightweight CNN that treats the log-mel spectrogram as a single-channel image.
    Input:  (B, 1, N_MELS, T)
    Output: (B, n_classes)
    """
    def __init__(self, n_classes: int, dropout: float = 0.4):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(1,   32,  pool=(2, 2)),
            ConvBlock(32,  64,  pool=(2, 2)),
            ConvBlock(64,  128, pool=(2, 2)),
            ConvBlock(128, 256, pool=(2, 2)),
        )
        # Global average pooling collapses the spatial dimensions
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)         # (B, 256, H', W')
        x = self.gap(x).flatten(1) # (B, 256)
        x = self.dropout(x)
        return self.head(x)        # (B, n_classes)


def get_pretrained_efficientnet(n_classes: int) -> nn.Module:
    """
    Alternative: fine-tune EfficientNet-B0 (better accuracy, more parameters).
    Requires: pip install torchvision
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # Adapt first conv to accept 1-channel input
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2,
                                     padding=1, bias=False)
    # Replace classifier head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model
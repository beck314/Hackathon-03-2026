import torch
import torch.nn as nn
import whisper


class WhisperBirdClassifier(nn.Module):
    def __init__(self, n_classes: int, model_size: str = "tiny", freeze_encoder: bool = True):
        super().__init__()

        # Load only the encoder from Whisper
        self.encoder = whisper.load_model(model_size).encoder
        print(self.encoder)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Whisper base encoder outputs (batch, time, 512)
        encoder_dim = self.encoder.ln_post.normalized_shape[0]

        self.head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 80, T) — log-mel spectrogram
        features = self.encoder(x)          # (B, T', encoder_dim)
        pooled   = features.mean(dim=1)     # (B, encoder_dim) — temporal average pool
        return self.head(pooled)            # (B, n_classes)
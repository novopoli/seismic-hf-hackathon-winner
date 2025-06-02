import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetWithFeatures(nn.Module):
    def __init__(self):
        super(UnetWithFeatures, self).__init__()

        # Encoder (in_channels увеличен до 2: сигнал + временной индекс)
        self.enc1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # Decoder
        self.dec1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.dec2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.dec3 = nn.Conv1d(16, 1, kernel_size=3, padding=1)

        # Преобразование координат через сверточный слой
        self.feature_conv = nn.Conv1d(3, 64, kernel_size=1)

        # Слой для объединения признаков и сигнала
        self.combine = nn.Conv1d(64 + 64, 64, kernel_size=1)

        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, features):
        # Добавим временной индекс как второй канал
        B, _, T = x.shape
        device = x.device
        time_index = torch.linspace(0, 1, steps=T, device=device).unsqueeze(0).unsqueeze(0)
        time_index = time_index.repeat(B, 1, 1)  # (B, 1, T)
        x = torch.cat([x, time_index], dim=1)    # (B, 2, T)

        # Encoder
        x1 = F.relu(self.enc1(x))
        x2 = self.pool(F.relu(self.enc2(x1)))
        x3 = self.pool(F.relu(self.enc3(x2)))  # (B, 64, T/4)

        # Преобразование координат (B, 3) → (B, 64, T/4)
        f = features.unsqueeze(-1)             # (B, 3, 1)
        f = self.feature_conv(f)               # (B, 64, 1)
        f = f.repeat(1, 1, x3.shape[2])        # (B, 64, T/4)

        # Комбинируем признаки и bottleneck
        x3_fused = self.combine(torch.cat([x3, f], dim=1))

        # Decoder
        x4 = self.upsample(F.relu(self.dec1(x3_fused)))
        if x4.size(2) != x2.size(2):
            x4 = F.pad(x4, (0, x2.size(2) - x4.size(2)))

        x5 = self.upsample(F.relu(self.dec2(x4 + x2)))
        if x5.size(2) != x1.size(2):
            x5 = F.pad(x5, (0, x1.size(2) - x5.size(2)))

        x6 = self.dec3(x5 + x1)
        return x6

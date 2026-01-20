import torch
import torch.nn as nn

# --- 1. CHANNEL ATTENTION ---
# "Fitur APA yang penting?"
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # Global Average Pooling & Global Max Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP (Multi-Layer Perceptron) bersama
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Jalur Average Pooling
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # Jalur Max Pooling
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # Gabungkan dan Sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)

# --- 2. SPATIAL ATTENTION ---
# "DI MANA fitur itu berada?"
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # Konvolusi 2 channel (hasil gabung Max & Avg) menjadi 1 channel mask
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Kompres channel jadi 1 lapis Max dan 1 lapis Avg
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Gabung (Concatenate)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # Konvolusi -> Sigmoid
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

# --- 3. BLOK CBAM UTAMA ---
class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7): # c1 = input channels
        super(CBAM, self).__init__()
        self.channel_gate = ChannelAttention(c1)
        self.spatial_gate = SpatialAttention(kernel_size)

    def forward(self, x):
        # Fitur dikalikan dengan bobot Channel
        x_out = self.channel_gate(x) * x
        # Hasilnya dikalikan dengan bobot Spatial
        x_out = self.spatial_gate(x_out) * x_out
        return x_out
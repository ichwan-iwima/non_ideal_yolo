import torch
import torch.nn as nn

class GlobalContextFusion(nn.Module):
    def __init__(self, c1, c2):
        """
        c1: Total Input Channels (Otomatis dari parser)
        c2: Output Channels (Otomatis dari parser atau YAML)
        """
        super().__init__()
        
        # --- LOGIKA AUTO-DETECT ---
        # Kita tahu Backbone YOLOv8n (SPPF) selalu output 256 channel.
        # Jadi, channel UTAMA adalah 256. Sisanya adalah KONTEKS.
        self.ch_main = 256 
        self.ch_context = c1 - self.ch_main
        
        print(f"\n✅ DEBUG FUSION: Total Input={c1} | Main={self.ch_main} | Context Detected={self.ch_context}")

        # Safety Check: Pastikan ada channel konteks
        if self.ch_context <= 0:
            raise ValueError(f"❌ Error: Input channel ({c1}) lebih kecil dari Main Feature (256). Cek urutan Concat!")

        # 1. Context Processor
        # Menerima sisa channel (berapapun jumlahnya, 16 atau 32)
        self.context_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(self.ch_context, 64, 1), # In: ch_context -> Out: 64
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.Sigmoid() 
        )
        
        # 2. Main Feature Processor
        self.main_conv = nn.Conv2d(self.ch_main, c2, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # x shape: [Batch, Total_Channels, H, W]
        
        # SPLIT TENSOR
        # Ambil 256 channel pertama sebagai Main
        x_main = x[:, :self.ch_main, :, :]
        
        # Ambil sisanya sebagai Context
        x_context = x[:, self.ch_main:, :, :] 
        
        # Fusi
        global_att = self.context_mlp(x_context)
        x_out = self.act(self.bn(self.main_conv(x_main)))
        
        # Kalikan fitur utama dengan attention dari konteks
        return x_out * global_att
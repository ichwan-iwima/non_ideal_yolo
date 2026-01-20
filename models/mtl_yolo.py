import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. HEAD KLASIFIKASI ILUMINASI ---
class IlluminationHead(nn.Module):
    def __init__(self, input_channels, num_classes=3):
        super().__init__()
        # Input dari backbone (C, H, W) -> Global Pool -> (C, 1, 1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes) # Output: 3 kelas (Low, Normal, Over)
        )

    def forward(self, x):
        # x shape: [Batch, Channel, Height, Width]
        # Ubah jadi [Batch, Channel, 1, 1] agar bisa masuk Linear layer
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.fc(x)

# --- 2. WRAPPER MULTI-TASK YOLO ---
class MTLYOLOWrapper(nn.Module):
    def __init__(self, original_yolo_model, lambda_illum=0.5):
        super().__init__()
        self.yolo = original_yolo_model
        self.lambda_illum = lambda_illum 
        
        # --- A. SETUP HOOK KE BACKBONE (FIXED) ---
        self.backbone_features = None
        
        def hook_fn(module, input, output):
            self.backbone_features = output

        # LOGIKA PENCARIAN LAYER YANG LEBIH AMAN
        # Kita cari container 'model' (Sequential)
        if hasattr(self.yolo, 'model'):
            # Ini struktur standar DetectionModel
            layers = self.yolo.model 
        else:
            # Fallback jika self.yolo itu sendiri sudah berupa Sequential
            layers = self.yolo
            
        # Pasang hook di layer ke-9 (Backbone End - SPPF biasanya)
        # Kita gunakan try-except agar tidak crash jika model lebih pendek
        try:
            target_layer = layers[9] 
            target_layer.register_forward_hook(hook_fn)
            # print("✅ Hook berhasil dipasang di Layer 9")
        except Exception as e:
            print(f"⚠️ Gagal pasang hook di Layer 9: {e}")
            # Fallback ke layer terakhir jika gagal (opsional)
            target_layer = layers[-1]
            target_layer.register_forward_hook(hook_fn)
        
        # Cek ukuran channel secara otomatis dengan Dummy Input
        # Ini penting agar Head kita tahu ukuran inputnya
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            # Kita jalankan forward pass singkat
            self.yolo(dummy) 
            
            if self.backbone_features is not None:
                in_channels = self.backbone_features.shape[1]
                print(f"⚡ MTL Init: Backbone channels detected = {in_channels}")
            else:
                # Fallback jika hook gagal total (jarang terjadi)
                in_channels = 512 
                print("⚠️ MTL Init: Hook gagal menangkap fitur, default channel 512")

        # --- B. BUAT HEAD BARU ---
        self.illum_head = IlluminationHead(input_channels=in_channels)
        
        # Sync Attributes
        self.args = {}
        if hasattr(original_yolo_model, 'nc'): self.nc = original_yolo_model.nc
        else: self.nc = 80
        
        # Copy names
        if hasattr(original_yolo_model, 'names'):
             self.names = original_yolo_model.names
        else:
             self.names = {i:str(i) for i in range(self.nc)}

    def forward(self, x, *args, **kwargs):
        # Sync Args
        if hasattr(self, 'args'):
            self.yolo.args = self.args

        # 1. Handle Input
        if isinstance(x, dict):
            img = x['img']
        else:
            img = x

        # 2. Jalankan YOLO 
        # (Hook akan otomatis mengisi self.backbone_features saat layer 9 dilewati)
        detection_output = self.yolo(x, *args, **kwargs)

        # 3. Jalankan Illumination Head
        # Pastikan fitur ada (safety check)
        if self.backbone_features is not None:
            illum_logits = self.illum_head(self.backbone_features)
        else:
            # Jika hook gagal, return dummy (untuk mencegah crash saat debug)
            # Seharusnya tidak masuk sini jika init sukses
            return detection_output

        # 4. TRAINING MODE: Hitung Loss Gabungan
        if self.training and isinstance(x, dict):
            # A. Label Otomatis (Heuristik Kecerahan)
            brightness = img.mean(dim=[1, 2, 3]) 
            illum_labels = torch.zeros_like(brightness, dtype=torch.long)
            illum_labels[brightness < 0.30] = 0        # Low
            illum_labels[brightness > 0.70] = 2        # Over
            illum_labels[(brightness >= 0.30) & (brightness <= 0.70)] = 1 # Normal
            
            # B. Loss Klasifikasi
            loss_illum = F.cross_entropy(illum_logits, illum_labels)
            
            # C. Gabung dengan Loss Deteksi
            # Output YOLO saat train biasanya: (loss_scalar, loss_items_tensor)
            if isinstance(detection_output, tuple):
                total_det_loss = detection_output[0]
                loss_items = detection_output[1]
                
                # Formula Multi-Task Loss
                total_loss = total_det_loss + (self.lambda_illum * loss_illum)
                
                return total_loss, loss_items
            else:
                # Fallback jika struktur return beda
                return detection_output + (self.lambda_illum * loss_illum)
            
        # 5. INFERENCE/VALIDATION MODE
        return detection_output

    # Safe Getattr
    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules: return modules[name]
        if '_parameters' in self.__dict__ and name in self.__dict__['_parameters']:
            return self.__dict__['_parameters'][name]
        if '_buffers' in self.__dict__ and name in self.__dict__['_buffers']:
            return self.__dict__['_buffers'][name]
        if 'yolo' in self.__dict__['_modules']:
            return getattr(self.__dict__['_modules']['yolo'], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
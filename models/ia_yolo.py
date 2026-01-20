import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. MODUL FILTER (Tetap) ---
class DifferentiableFilters(nn.Module):
    def __init__(self):
        super().__init__()

    def adjust_gamma(self, img, gamma_param):
        gamma = gamma_param * 4.9 + 0.1 
        return torch.pow(img + 1e-6, gamma)

    def adjust_exposure(self, img, exp_param):
        exp = (exp_param - 0.5) * 4 
        return img * torch.exp(exp * 0.693)

    def adjust_contrast(self, img, cont_param):
        alpha = cont_param + 0.5
        mean_lum = torch.mean(img, dim=[2, 3], keepdim=True)
        return torch.clamp((img - mean_lum) * alpha + mean_lum, 0, 1)

    def forward(self, img, params):
        x = self.adjust_gamma(img, params[:, 0].view(-1, 1, 1, 1))
        x = self.adjust_exposure(x, params[:, 1].view(-1, 1, 1, 1))
        x = self.adjust_contrast(x, params[:, 2].view(-1, 1, 1, 1))
        return x

# --- 2. MODUL DIP (Tetap) ---
class DIP_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.InstanceNorm2d(16), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 3), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        # Resize input DIP ke ukuran kecil (128x128)
        x_small = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        return self.net(x_small)

# --- 3. WRAPPER UTAMA IA-YOLO (UPDATED WITH ARGS SYNC) ---
class IAYOLOWrapper(nn.Module):
    def __init__(self, original_yolo_model):
        super().__init__()
        self.dip = DIP_Network()
        self.filters = DifferentiableFilters()
        self.yolo = original_yolo_model
        
        # Copy Attributes Awal
        if hasattr(original_yolo_model, 'nc'):
            self.nc = original_yolo_model.nc
        elif hasattr(original_yolo_model, 'yaml') and isinstance(original_yolo_model.yaml, dict):
            self.nc = original_yolo_model.yaml.get('nc', 80)
        else:
            self.nc = 80 
            
        if hasattr(original_yolo_model, 'names'):
            self.names = original_yolo_model.names
        elif hasattr(original_yolo_model, 'yaml') and 'names' in original_yolo_model.yaml:
            self.names = original_yolo_model.yaml['names']
        else:
            self.names = {i: str(i) for i in range(self.nc)}

        # Inisialisasi args kosong dulu
        self.args = {} 

    def forward(self, x, *args, **kwargs):
        # --- PERBAIKAN KRUSIAL: SINKRONISASI ARGS ---
        # Trainer Ultralytics menempelkan 'args' ke Wrapper ini.
        # Kita harus meneruskannya ke self.yolo agar fungsi Loss tidak crash.
        if hasattr(self, 'args'):
            self.yolo.args = self.args

        # 1. Handle Input (Dict vs Tensor)
        if isinstance(x, dict):
            img = x['img']
        else:
            img = x
            
        # 2. IA-YOLO Step: Enhance Gambar
        params = self.dip(img)
        img_enhanced = self.filters(img, params)
        
        # 3. Pass to YOLO
        if isinstance(x, dict):
            x['img'] = img_enhanced
            return self.yolo(x, *args, **kwargs)
        else:
            return self.yolo(img_enhanced, *args, **kwargs)

    # --- Safe Getattr (Tetap diperlukan) ---
    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
                
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters: return _parameters[name]
            
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers: return _buffers[name]

        if '_modules' in self.__dict__ and 'yolo' in self.__dict__['_modules']:
            return getattr(self.__dict__['_modules']['yolo'], name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
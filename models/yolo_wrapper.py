import torch
import torch.nn as nn
from ultralytics import YOLO
from models.dip_module import DIP_Network, DifferentiableFilters

class IA_YOLO(nn.Module):
    def __init__(self, yolo_weights='yolov8n.pt'):
        super().__init__()
        self.dip = DIP_Network()
        self.filters = DifferentiableFilters()
        
        # Load pre-trained YOLO model backbone only
        # Kita load full model dulu, nanti kita ambil bagian feature extractionnya
        self.yolo_model = YOLO(yolo_weights).model 
        
    def forward(self, x):
        # 1. Prediksi Parameter Cahaya
        params = self.dip(x)
        
        # 2. Perbaiki Gambar (Enhance)
        x_enhanced = self.filters(x, params)
        
        # 3. Masukkan gambar yang sudah diperbaiki ke YOLO
        # Return format YOLO standard
        preds = self.yolo_model(x_enhanced)
        
        if self.training:
             return preds # Saat training return loss/prediction structure
        else:
             return preds, x_enhanced, params # Saat inferensi kita butuh lihat hasil editnya
from ultralytics import YOLO
# GANTI IMPORT: Jangan pakai BaseTrainer, tapi pakai DetectionTrainer
from ultralytics.models.yolo.detect import DetectionTrainer 
from models.mtl_yolo import MTLYOLOWrapper
import torch
import torch.nn as nn

# --- CUSTOM TRAINER (Inherit dari DetectionTrainer) ---
class MTLYOLOTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        # --- LOGIKA LOAD MODEL (Sama seperti fix sebelumnya) ---
        
        yolo_model = None
        
        # Kasus 1: 'weights' adalah Object Model yang sudah di-load
        if weights and isinstance(weights, nn.Module):
            yolo_model = weights
        
        # Kasus 2: 'weights' adalah String Path atau None
        else:
            model_path = weights if weights else self.args.model
            yolo_model = YOLO(model_path).model
        
        # --- WRAPPING ---
        # Cek double wrap
        if isinstance(yolo_model, MTLYOLOWrapper):
            return yolo_model
            
        # Bungkus dengan Wrapper MTL
        # lambda_illum=0.5 (Bobot loss iluminasi)
        model = MTLYOLOWrapper(yolo_model, lambda_illum=0.5) 
        
        if verbose:
            print(f"✅ Model Multi-Task Learning (MTL) siap!")
            print("   - Backbone Shared")
            print("   - Cabang Deteksi: Standar YOLO")
            print("   - Cabang Iluminasi: GlobalAvgPool -> Linear (3 Kelas)")
            
        return model

# --- MAIN EXECUTION ---
def start_training():
    # Setup Argument
    args = dict(
        model='yolov8n.pt', 
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='project_mtl_yolo', 
        name='experiment_2_mtl',
        device='cpu', 
        optimizer='AdamW'
    )
    
    trainer = MTLYOLOTrainer(overrides=args)
    trainer.train()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    start_training()
from ultralytics import YOLO
from models.fusion_modules import GlobalContextFusion # Import modul baru
import ultralytics.nn.tasks as tasks
import torch

# --- REGISTER MODULE ---
tasks.GlobalContextFusion = GlobalContextFusion

def start_training():
    # 1. Load Model dari YAML Fusion
    model = YOLO("models/yolov8_fusion.yaml") 
    
    # 2. Transfer Weights
    # Penting: Karena struktur berubah cukup drastis di layer 10,
    # kita load weights yolov8n.pt tapi biarkan layer baru terinisialisasi random.
    try:
        model = model.load("yolov8n.pt")
    except:
        print("⚠️ Info: Beberapa weight mungkin tidak cocok karena struktur baru. Ini normal.")

    # 3. Setup Arguments
    args = dict(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='project_fusion_yolo',
        name='experiment_4_fusion',
        device='cpu', 
        optimizer='AdamW',
        lr0=0.001
    )
    
    # 4. Train
    print("🚀 Mulai Training dengan Feature-Level Fusion...")
    model.train(**args)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    start_training()
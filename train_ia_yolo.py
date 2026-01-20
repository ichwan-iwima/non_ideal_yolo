import sys
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics import YOLO
from models.ia_yolo import IAYOLOWrapper
import torch

# 1. Buat Custom Trainer Class
class IAYOLOTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Fungsi ini dipanggil otomatis oleh YOLO saat inisialisasi.
        Kita override agar me-load model kita (IAYOLOWrapper), bukan YOLO standar.
        """
        # Load model YOLO standar dulu (misal yolov8n.pt)
        # Kita gunakan method parent class untuk load struktur dasar
        yolo_model = super().get_model(cfg, weights, verbose)
        
        # Bungkus dengan IA-YOLO
        print("⚡ MENGAKTIFKAN MODUL IA-YOLO (DIP + FILTERS) ⚡")
        model = IAYOLOWrapper(yolo_model)
        
        # Pastikan parameter DIP masuk ke device yang benar (GPU/CPU)
        return model

# 2. Fungsi Main Training
def start_training():
    # Load hyperparams dasar dari file konfigurasi
    # (Pastikan data.yaml Anda sudah benar nc dan names-nya)
    
    args = dict(
        model='yolov8n.pt', # Base weights
        data='data.yaml',   # Path ke data config
        epochs=50,          # Jumlah epoch
        imgsz=640,
        batch=4,            # Sesuaikan VRAM
        device='cuda' if torch.cuda.is_available() else 'cpu',           # 0 untuk GPU, 'cpu' untuk CPU
        project='project_ia_yolo',
        name='experiment_1_ia_yolo',
        
        # Hyperparam penting untuk Fine-Tuning
        lr0=1e-3,           # Learning rate awal
        optimizer='AdamW'   # AdamW biasanya bagus untuk custom net
    )

    # Inisialisasi Trainer Kustom kita
    trainer = IAYOLOTrainer(overrides=args)
    
    # Mulai Training
    trainer.train()

if __name__ == '__main__':
    start_training()
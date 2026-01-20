from ultralytics import YOLO
from models.attention_modules import CBAM # Import modul kita
import ultralytics.nn.tasks as tasks
import torch

# --- HACK: REGISTER CUSTOM MODULE ---
# Kita tambahkan CBAM ke dalam dictionary global module Ultralytics
# agar saat membaca YAML, dia tahu apa itu 'CBAM'
tasks.CBAM = CBAM 

def start_training():
    # 1. Load Model dari YAML Custom
    # Ini akan membangun model baru dari nol (random weights)
    # namun struktur sesuai YAML yang kita buat
    model = YOLO("models/yolov8_cbam.yaml") 
    
    # 2. Transfer Weights (Opsional tapi DISARANKAN)
    # Karena model dibangun dari nol, performanya jelek di awal.
    # Kita bisa load weight dari yolov8n.pt standar untuk layer yang namanya sama (Backbone awal).
    # Ultralytics biasanya melakukan ini otomatis jika nama filenya berakhiran .yaml,
    # dia akan coba cari pretrained weights yolov8n.pt untuk inisialisasi layer yang cocok.
    model = model.load("yolov8n.pt") 

    # 3. Setup Arguments
    args = dict(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='project_cbam_yolo',
        name='experiment_3_cbam',
        device='cpu', # Ubah ke 0 jika pakai GPU
        optimizer='AdamW',
        lr0=0.001 # Learning rate sedikit lebih kecil karena struktur berubah
    )
    
    # 4. Train
    print("🚀 Mulai Training dengan Arsitektur CBAM...")
    model.train(**args)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    start_training()
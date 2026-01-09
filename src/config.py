import torch
import os

class Config:
    DATA_DIR = 'data'
    CSV_FILE = os.path.join(DATA_DIR, 'flickr8k_training_data.csv')
    IMG_DIR = os.path.join(DATA_DIR, 'flickr8k_dataset', 'Images')
    GLOVE_DIR = '.vector_cache'
    
    # Parametry Modelu
    EMBED_DIM = 300
    HIDDEN_DIM = 512
    VISUAL_EMBED_DIM = 512
    
    # Parametry Treningu
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 15
    NUM_WORKERS = 4
    
    # Obraz
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Urządzenie
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Konfiguracja załadowana. Urządzenie: {Config.DEVICE}")
import torch
import os

CSV_FILE = 'flickr8k_training_data.csv'  # Plik wygenerowany wcześniej
MODEL_SAVE_PATH = 'best_model.pth'

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_LEN = 64
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config
from src.utils import load_glove_embeddings
from src.dataset import FlickrDataset, Vocabulary, MyCollate
from src.model import CrossModalNetwork
from src.train import train_epoch, evaluate

def main():
    # download_spacy_model()
    
    # 2. Transformacje
    transforms_train = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(Config.MEAN, Config.STD)
    ])

    print("Budowanie słownika...")
    if not os.path.exists(Config.CSV_FILE):
        raise FileNotFoundError(f"Brak pliku CSV w {Config.CSV_FILE}")
    
    raw_df = pd.read_csv(Config.CSV_FILE)
    all_captions = raw_df['caption'].tolist()
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(all_captions)
    print(f"Rozmiar słownika: {len(vocab)}")

    print("Zapisywanie słownika do vocab.pth (dla submission)...")
    torch.save({'stoi': vocab.stoi}, "vocab.pth")

    dataset = FlickrDataset(
        csv_file=Config.CSV_FILE,
        root_dir=Config.IMG_DIR,
        vocab=vocab,
        transform=transforms_train
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    pad_idx = vocab.stoi["<PAD>"]
    train_loader = DataLoader(
        train_set, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS, 
        collate_fn=MyCollate(pad_idx),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS, 
        collate_fn=MyCollate(pad_idx),
        pin_memory=True
    )

    # 5. Model
    glove_matrix = load_glove_embeddings(vocab, embedding_dim=Config.EMBED_DIM)
    
    model = CrossModalNetwork(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        visual_dim=Config.VISUAL_EMBED_DIM,
        pretrained_embeddings=glove_matrix
    ).to(Config.DEVICE)

    # 6. Setup treningu
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # 7. Pętla
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    print(f"Rozpoczynam trening na {Config.DEVICE}...")
    
    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.1e}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights.pth")
            print(f"--> Zapisano model (Acc: {best_acc:.4f})")

    plt.plot(history['train_loss'], label='Loss')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.savefig('training_history.png')
    print("Zakończono. Wykres zapisany jako training_history.png")

import os
if __name__ == "__main__":
    main()
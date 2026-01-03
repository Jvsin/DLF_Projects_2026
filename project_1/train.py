import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

import config
from dataset import ImageTextDataset, get_transforms
from model import VerificationModel

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch in loop:
        images = batch['image'].to(config.DEVICE)
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        labels = batch['label'].to(config.DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_description(f"Loss: {loss.item():.4f}")
        
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(config.DEVICE)
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE).unsqueeze(1)
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / len(loader), correct / total

def main():
    print(f"Uruchamianie treningu na: {config.DEVICE}")
    
    if not os.path.exists(config.CSV_FILE):
        print(f"BŁĄD: Nie znaleziono {config.CSV_FILE}. Uruchom najpierw generator danych!")
        return

    full_dataset = ImageTextDataset(config.CSV_FILE, transform=get_transforms())
    
    #dataset split 90/10
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"Liczba przykładów treningowych: {len(train_dataset)}")
    print(f"Liczba przykładów walidacyjnych: {len(val_dataset)}")
    
    #model define
    model = VerificationModel().to(config.DEVICE)
    
    #optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoka {epoch+1}/{config.EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        #save model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("Best model saved")

    print("\nTrainign finished..")

if __name__ == "__main__":
    main()
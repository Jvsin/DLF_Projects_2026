import torch
from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # tqdm z file=None domyślnie leci na stderr, co jest ok w terminalu
    loop = tqdm(loader, leave=False, desc="Training")
    
    for imgs, caps, labels in loop:
        imgs = imgs.to(device)
        caps = caps.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(imgs, caps)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        loop.set_description(f"Loss: {loss.item():.4f}")
        
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, caps, labels in loader:
            imgs = imgs.to(device)
            caps = caps.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(imgs, caps)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return total_loss / len(loader), correct / total
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import os
import re

class SubmissionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Konfiguracja (Musi pasować do Twojego treningu!)
        self.embed_dim = 300
        self.hidden_dim = 512
        self.visual_dim = 512
        
        # 2. Ładowanie słownika z pliku w ZIPie
        # Szukamy pliku vocab.pth w tym samym folderze co model.py
        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.pth')
        
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Nie znaleziono pliku: {vocab_path}")
            
        # Wczytujemy słownik
        vocab_data = torch.load(vocab_path, map_location='cpu', weights_only=False)
        
        # Obsługa różnych formatów zapisu (na wszelki wypadek)
        if isinstance(vocab_data, dict) and 'stoi' in vocab_data:
            self.stoi = vocab_data['stoi']
        else:
            self.stoi = vocab_data # Zakładamy, że to bezpośrednio dict
            
        self.vocab_size = len(self.stoi)
        
        # =====================================================================
        # 3. ODTWORZENIE ARCHITEKTURY (CrossModalNetwork)
        # Nazwy zmiennych muszą być IDENTYCZNE jak w weights.pth
        # =====================================================================
        
        # A. ResNet
        resnet = models.resnet50(weights=None) # Internet wyłączony, wagi wczytają się z pliku
        modules = list(resnet.children())[:-1] 
        self.resnet_base = nn.Sequential(*modules)
        
        # B. Projekcja Obrazu
        self.image_projection = nn.Sequential(
            nn.Linear(2048, self.visual_dim),
            nn.BatchNorm1d(self.visual_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # C. Embedding i LSTM
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        
        # D. Projekcja Tekstu
        self.text_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.visual_dim),
            nn.BatchNorm1d(self.visual_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # E. Klasyfikator
        self.classifier = nn.Sequential(
            nn.Linear(self.visual_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, images, captions):
        """Logika przepływu (Forward Pass)"""
        # Obraz
        img_features = self.resnet_base(images).squeeze(-1).squeeze(-1)
        img_emb = self.image_projection(img_features)
        
        # Tekst
        embeds = self.embedding(captions)
        lstm_out, _ = self.lstm(embeds)
        
        # Mean Pooling
        txt_features = torch.mean(lstm_out, dim=1) 
        txt_emb = self.text_projection(txt_features)
        
        # FUZJA: Mnożenie element-wise (zgodnie z Twoim kodem)
        fused_vector = img_emb * txt_emb
        
        output = self.classifier(fused_vector)
        return output

    def _tokenizer_eng(self, text):
        """Prosty tokenizer (bez spacy)"""
        text = str(text).lower()
        return re.findall(r"[\w]+|[^\s\w]", text)

    def _numericalize(self, text):
        """Zamiana tekstu na liczby"""
        tokens = self._tokenizer_eng(text)
        # SOS=1, EOS=2, UNK=3 (Zgodnie z Twoim dataset.py)
        indices = [1] 
        for token in tokens:
            indices.append(self.stoi.get(token, 3)) 
        indices.append(2) 
        return torch.tensor(indices, dtype=torch.long)

    def predict(self, image_tensor, text_string):
        """
        Metoda wymagana przez system submission.
        """
        self.eval()
        with torch.no_grad():
            # 1. Dodajemy wymiar batcha do obrazu: (C, H, W) -> (1, C, H, W)
            img_batch = image_tensor.unsqueeze(0)
            
            # 2. Tekst na tensor -> (1, Seq_Len)
            text_tensor = self._numericalize(text_string).unsqueeze(0)
            
            # 3. Urządzenie (CPU/GPU)
            device = img_batch.device
            text_tensor = text_tensor.to(device)
            
            # 4. Predykcja
            logits = self.forward(img_batch, text_tensor)
            prob = torch.sigmoid(logits)
            
            return prob.item()

# Funkcja transformacji (Domyślna jest OK, ale dla pewności dodajemy)
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
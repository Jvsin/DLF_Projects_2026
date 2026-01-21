import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from collections import Counter
from .config import Config

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        text = text.lower()
        return re.findall(r"[\w]+|[^\s\w]", text)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(self, csv_file, vocab, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab
        
        self.positives = self.df[self.df['label'] == 1].reset_index(drop=True)
        self.negatives = self.df[self.df['label'] == 0].reset_index(drop=True)
        self.dataset_len = len(self.positives) * 2

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if idx % 2 == 0:
            pos_idx = (idx // 2) % len(self.positives)
            row = self.positives.iloc[pos_idx]
            label = 1.0
        else:
            row = self.negatives.sample(n=1).iloc[0]
            label = 0.0

        img_path_raw = row['image_path']
        # Obsługa ścieżek Windows/Linux
        img_name = os.path.basename(img_path_raw.replace('\\', '/'))
        img_path = os.path.join(self.root_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback na czarny obraz w razie błędu pliku
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), 'black')

        if self.transform:
            image = self.transform(image)

        caption_vec = [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(row['caption'])
        caption_vec.append(self.vocab.stoi["<EOS>"])

        return {
            "image": image,
            "caption": torch.tensor(caption_vec),
            "label": torch.tensor(label, dtype=torch.float)
        }

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item['image'].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item['caption'] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        labels = torch.tensor([item['label'] for item in batch])
        return imgs, targets, labels
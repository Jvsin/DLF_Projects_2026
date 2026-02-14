import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import re
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return re.findall(r"[\w]+|[^\s\w]", str(text).lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        print("Budowanie słownika...")
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        print(f"Rozmiar słownika: {len(self.itos)} słów")

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(self, csv_file, vocab, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab
        
        if 'label' not in self.df.columns:
            raise ValueError("Plik CSV musi zawierać kolumnę 'label' (0 lub 1)!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_name = os.path.basename(row['image_path'])
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback dla uszkodzonych/brakujących zdjęć (żeby nie wywaliło treningu)
            image = Image.new('RGB', (224, 224), 'black')

        if self.transform:
            image = self.transform(image)

        caption = row['caption']
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        label = float(row['label'])

        return {
            'image': image,
            'caption': torch.tensor(numericalized_caption),
            'label': torch.tensor(label, dtype=torch.float)
        }

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item['image'].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item['caption'] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)

        return imgs, targets, labels
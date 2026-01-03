import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import DistilBertTokenizer
from torchvision import transforms
import pandas as pd
import config

class ImageTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # Ładujemy tokenizator (cache'owany lokalnie po pierwszym uruchomieniu)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, FileNotFoundError):
            image = Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE), color='black')
            
        if self.transform:
            image = self.transform(image)

        #text tokenization
        caption = str(row['caption'])
        tokens = self.tokenizer(
            caption,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        #labels
        label = torch.tensor(row['label'], dtype=torch.float)

        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': label
        }

def get_transforms():
    """Zwraca standardowe transformacje dla ResNet"""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
    ])
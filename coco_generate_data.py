import spacy
import random
import json
import pandas as pd
import os
from tqdm import tqdm  # Pasek postępu

BASE_DIR = 'coco_dataset'
JSON_PATH = os.path.join(BASE_DIR, 'annotations_trainval2017', 'annotations', 'captions_train2017.json')
IMAGES_DIR = os.path.join(BASE_DIR, 'train2017', 'train2014')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Pobieranie modelu spacy...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DataAdversary:
    def __init__(self):
        self.colors = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "gray", "pink"]
        self.numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        self.numbers_digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        
    def replace_word(self, text, original_word, possible_replacements):
        choices = [w for w in possible_replacements if w != original_word.lower()]
        if not choices: return text
        new_word = random.choice(choices)
        return text.replace(original_word, new_word)

    def generate_hard_negative(self, caption):
        doc = nlp(caption)
        original_text = caption
        
        # Zmiana kolorów
        for token in doc:
            if token.text.lower() in self.colors:
                new_text = self.replace_word(original_text, token.text, self.colors)
                return new_text, "color_swap"

        # Zmiana liczebników
        for token in doc:
            if token.text.lower() in self.numbers:
                new_text = self.replace_word(original_text, token.text, self.numbers)
                return new_text, "number_swap"
        
        # Zmiana cyfr
        for token in doc:
            if token.text in self.numbers_digits:
                new_text = self.replace_word(original_text, token.text, self.numbers_digits)
                return new_text, "digit_swap"
        
        return None, None

def create_dataset(json_path, images_dir, limit=50000):
    adversary = DataAdversary()
    dataset = []
    
    print(f"Wczytywanie JSON: {json_path}")
    with open(json_path, 'r') as f:
        coco_raw = json.load(f)
    
    # Lista wszystkich opisów do losowania Easy Negatives
    all_captions = [x['caption'] for x in coco_raw['annotations']]
    
    print(f"Przetwarzanie danych (Limit: {limit})...")
    
    processed_count = 0
    
    # Używamy tqdm do paska postępu
    for ann in tqdm(coco_raw['annotations']):
        if processed_count >= limit: break
        
        image_id = ann['image_id']
        caption = ann['caption']
        
        # --- TWORZENIE NAZWY PLIKU (Naprawa problemu) ---
        # Format: COCO_train2014_000000xxxxxx.jpg
        filename = f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
        full_path = os.path.join(images_dir, filename)
        
        # Sprawdzamy czy plik istnieje. Jeśli nie - pomijamy ten przykład.
        if not os.path.exists(full_path):
            continue
            
        processed_count += 1
        
        # 1. Pozytyw
        dataset.append({
            'image_path': full_path,
            'caption': caption,
            'label': 1,
            'type': 'positive'
        })
        
        # 2. Hard Negative
        fake_caption, change_type = adversary.generate_hard_negative(caption)
        if fake_caption:
            dataset.append({
                'image_path': full_path,
                'caption': fake_caption,
                'label': 0,
                'type': f'hard_neg_{change_type}'
            })
            
        # 3. Easy Negative
        while True:
            random_caption = random.choice(all_captions)
            if random_caption != caption:
                dataset.append({
                    'image_path': full_path,
                    'caption': random_caption,
                    'label': 0,
                    'type': 'easy_neg'
                })
                break
    
    return pd.DataFrame(dataset)

if __name__ == "__main__":
    # Uruchomienie generowania
    if os.path.exists(JSON_PATH) and os.path.exists(IMAGES_DIR):
        df = create_dataset(JSON_PATH, IMAGES_DIR, limit=50000)
        
        if not df.empty:
            output_file = 'train_dataset_final.csv'
            df.to_csv(output_file, index=False)
            print(f"\nSukces! Zapisano {len(df)} przykładów do pliku: {output_file}")
            print(df['type'].value_counts())
        else:
            print("\nBłąd: Wygenerowano pusty DataFrame. Sprawdź czy zdjęcia na pewno są w folderze.")
    else:
        print(f"\Błąd ścieżek:\nJSON: {os.path.exists(JSON_PATH)}\nIMG: {os.path.exists(IMAGES_DIR)}")
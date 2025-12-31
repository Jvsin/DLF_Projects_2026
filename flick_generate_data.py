import pandas as pd
import spacy
import random
import os
from tqdm import tqdm


BASE_DIR = 'flickr8k_dataset'
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
CAPTIONS_FILE = os.path.join(BASE_DIR, 'captions.txt')

OUTPUT_FILE = 'flickr8k_training_data.csv'

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Pobieranie modelu spacy...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class DataAdversary:
    def __init__(self):
        self.colors = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown"]
        self.numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        self.numbers_digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        self.people_map = {
            "boy": "girl", "girl": "boy",
            "man": "woman", "woman": "man",
            "dog": "cat", "cat": "dog"
        }

    def replace_word(self, text, original_word, possible_replacements):
        choices = [w for w in possible_replacements if w != original_word.lower()]
        if not choices: return text
        new_word = random.choice(choices)
        return text.replace(original_word, new_word)

    def generate_hard_negative(self, caption):
        doc = nlp(caption.lower()) 
        original_text = caption
        
        # Zamiana kolorów
        for token in doc:
            if token.text in self.colors:
                return self.replace_word(original_text, token.text, self.colors), "color_swap"

        # Zamiana ilości
        for token in doc:
            if token.text in self.numbers:
                return self.replace_word(original_text, token.text, self.numbers), "number_swap"
        
        # Zamiana cyfr
        for token in doc:
            if token.text in self.numbers_digits:
                return self.replace_word(original_text, token.text, self.numbers_digits), "digit_swap"
                
        # Zamiana przedmiotów / ludzi / zwierząt
        for token in doc:
            if token.text in self.people_map:
                new_word = self.people_map[token.text]
                return original_text.replace(token.text, new_word), "entity_swap"

        return None, None

def process_flickr():
    print(f"Wczytywanie: {CAPTIONS_FILE}")
    
    try:
        df_raw = pd.read_csv(CAPTIONS_FILE)
    except Exception as e:
        print(f"Błąd odczytu CSV: {e}")
        return

    adversary = DataAdversary()
    dataset = []
    
    all_captions = df_raw['caption'].dropna().tolist()
    
    print(f"Przetwarzanie {len(df_raw)} wierszy...")

    for index, row in tqdm(df_raw.iterrows(), total=len(df_raw)):
        image_name = row['image']
        caption = str(row['caption'])
        
        full_image_path = os.path.join(IMAGES_DIR, image_name)
        
        if not os.path.exists(full_image_path):
            continue

        dataset.append({
            'image_path': full_image_path,
            'caption': caption,
            'label': 1,
            'type': 'positive'
        })

        # Generowanie negatywów hard
        fake_caption, error_type = adversary.generate_hard_negative(caption)
        
        if fake_caption:
            dataset.append({
                'image_path': full_image_path,
                'caption': fake_caption,
                'label': 0,
                'type': f'hard_neg_{error_type}'
            })
        else:
            pass

        # Easy negative - losowy błędny opis z innego obrazka
        while True:
            random_caption = random.choice(all_captions)
            if random_caption != caption:
                dataset.append({
                    'image_path': full_image_path,
                    'caption': random_caption,
                    'label': 0,
                    'type': 'easy_neg'
                })
                break
    
    final_df = pd.DataFrame(dataset)
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Zapisano plik: {OUTPUT_FILE}")
    print(f"Lączna liczba par treningowych: {len(final_df)}")
    print(final_df['type'].value_counts())

if __name__ == "__main__":
    if os.path.exists(IMAGES_DIR) and os.path.exists(CAPTIONS_FILE):
        process_flickr()
    else:
        print("Nie znaleziono folderu Images lub pliku captions.txt.")
        print(f"Sprawdź ścieżkę: {BASE_DIR}")
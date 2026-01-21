# prepare_submission.py
import torch
import pandas as pd
import os
from src.config import Config
from src.dataset import Vocabulary

MODEL_SOURCE = "best_model_local_2.pth"  # Twój najlepszy model
weights_dest = "submission/weights.pth"
vocab_dest = "submission/vocab.pth"

os.makedirs("submission", exist_ok=True)

# 2. Generowanie Słownika (IDENTYCZNIE JAK W TRENINGU)
print("Generowanie słownika z CSV...")
if not os.path.exists(Config.CSV_FILE):
    raise FileNotFoundError("Brak pliku CSV! Uruchom to tam, gdzie trenowałeś.")

raw_df = pd.read_csv(Config.CSV_FILE)
all_captions = raw_df['caption'].tolist()
vocab = Vocabulary(freq_threshold=2)
vocab.build_vocabulary(all_captions)

print(f"Słownik gotowy. Rozmiar: {len(vocab)}")

# 3. Zapisywanie słownika
# Zapisujemy sam słownik mapujący (stoi) i odwrotny (itos)
vocab_data = {
    'stoi': vocab.stoi,
    'itos': vocab.itos
}
torch.save(vocab_data, vocab_dest)
print(f"Słownik zapisany do: {vocab_dest}")

print("Kopiowanie wag...")

state_dict = torch.load(MODEL_SOURCE, map_location='cpu')
torch.save(state_dict, weights_dest)
print(f"Wagi zapisane do: {weights_dest}")

print("\nGOTOWE! Twoje pliki do ZIP to:")
print(f"1. submission/model.py (Ten musisz stworzyć z kodu poniżej)")
print(f"2. {weights_dest}")
print(f"3. {vocab_dest}")
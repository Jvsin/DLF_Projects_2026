import pandas as pd
import os

CSV_FILE = 'flickr8k_training_data.csv'

if not os.path.exists(CSV_FILE):
    print(f"Plik {CSV_FILE} nie istnieje!")
    exit(1)

print(f"Wczytywanie pliku: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

print(f"Liczba wierszy przed czyszczeniem: {len(df)}")
print(f"Przykładowe opisy PRZED:\n{df['caption'].head()}\n")

# Usuwanie kropek z końca opisów
df['caption'] = df['caption'].str.rstrip('.')

print(f"Przykładowe opisy PO:\n{df['caption'].head()}\n")

# Zapisywanie poprawionego pliku
df.to_csv(CSV_FILE, index=False)

print(f"✓ Plik {CSV_FILE} został zaktualizowany!")
print(f"Liczba wierszy po czyszczeniu: {len(df)}")

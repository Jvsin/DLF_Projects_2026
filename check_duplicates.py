import pandas as pd
from collections import defaultdict

# Wczytaj dane
df = pd.read_csv('data/flickr8k_training_data.csv')

print(f"Całkowita liczba wierszy: {len(df)}")
print(f"Liczba unikalnych obrazków: {df['image_path'].nunique()}")
print()

# Sprawdzenie duplikatów (ten sam obraz + ten sam opis)
duplicates = defaultdict(list)

for idx, row in df.iterrows():
    key = (row['image_path'], row['caption'])
    duplicates[key].append({
        'index': idx,
        'label': row['label'],
        'type': row['type']
    })

# Znajdź duplikaty z różnymi etykietami
problematic_duplicates = []
for key, entries in duplicates.items():
    if len(entries) > 1:
        labels = [e['label'] for e in entries]
        # Sprawdź czy są różne etykiety
        if len(set(labels)) > 1:
            problematic_duplicates.append((key, entries))

# Wyświetl wyniki
if problematic_duplicates:
    print(f"⚠️  ZNALEZIONO {len(problematic_duplicates)} PRZYPADKÓW DUPLIKATÓW Z RÓŻNYMI ETYKIETAMI!\n")
    print("="*100)
    
    for i, ((image_path, caption), entries) in enumerate(problematic_duplicates, 1):
        print(f"\n{i}. Obrazek: {image_path}")
        print(f"   Opis: {caption}")
        print(f"   Liczba wystąpień: {len(entries)}")
        print(f"   Szczegóły:")
        for entry in entries:
            print(f"      - Wiersz {entry['index']}: label={entry['label']}, type={entry['type']}")
        print("-"*100)
else:
    print("✓ Nie znaleziono duplikatów z różnymi etykietami")

# Dodatkowe statystyki
print("\n" + "="*100)
print("DODATKOWE STATYSTYKI:")
print("="*100)

# Wszystkie duplikaty (nawet z tymi samymi etykietami)
all_duplicates = [key for key, entries in duplicates.items() if len(entries) > 1]
if all_duplicates:
    print(f"\nWszystkie duplikaty (obraz + opis): {len(all_duplicates)}")
    print("Przykłady:")
    for i, (image_path, caption) in enumerate(all_duplicates[:5], 1):
        entries = duplicates[(image_path, caption)]
        labels = [e['label'] for e in entries]
        print(f"  {i}. {image_path}")
        print(f"     Opis: {caption[:60]}...")
        print(f"     Wystąpienia: {len(entries)}, Etykiety: {labels}")

# Sprawdź czy są identyczne wiersze (wszystkie kolumny takie same)
full_duplicates = df[df.duplicated(keep=False)]
if len(full_duplicates) > 0:
    print(f"\n⚠️  Identyczne wiersze (wszystkie kolumny): {len(full_duplicates)}")
    print(full_duplicates.head(10))
else:
    print(f"\n✓ Brak identycznych wierszy (wszystkie kolumny)")

# Statystyki według typu
print("\n" + "="*100)
print("ROZKŁAD TYPÓW:")
print("="*100)
print(df['type'].value_counts())

print("\n" + "="*100)
print("ROZKŁAD ETYKIET:")
print("="*100)
print(df['label'].value_counts())

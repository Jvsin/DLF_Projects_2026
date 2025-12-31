import os
import json

# Ustaw ścieżki zgodnie z Twoją strukturą folderów
base_dir = 'coco_dataset' 
json_path = os.path.join(base_dir, 'annotations_trainval2017', 'annotations', 'captions_train2017.json')
imgs_dir = os.path.join(base_dir, 'train2017', 'train2014')

# 1. Sprawdźmy czy pliki istnieją
print(f"Szukam JSONa w: {json_path}")
print(f"Szukam zdjęć w: {imgs_dir}")

if os.path.exists(json_path) and os.path.exists(imgs_dir):
    print("Ścieżki folderów są poprawne! 👍")
    
    # 2. Sprawdźmy przykładowy plik zdjęcia
    files = os.listdir(imgs_dir)
    if files:
        sample_file = files[0]
        print(f"Przykładowy plik zdjęcia: {sample_file}")
        
        # 3. Sprawdźmy co jest w JSONie
        with open(json_path, 'r') as f:
            data = json.load(f)
            sample_ann = data['annotations'][0]
            img_id = sample_ann['image_id']
            print(f"Przykładowe ID w JSON: {img_id}")
            
            # Weryfikacja dopasowania
            # Sprawdzamy czy ID (np. 123) jest w nazwie pliku
            if str(img_id) in sample_file:
                print("Dopasowanie ID -> Plik wygląda OK.")
            else:
                print("UWAGA: ID z JSONa nie pasuje bezpośrednio do nazwy pliku. Będziemy musieli to obsłużyć.")
    else:
        print("BŁĄD: Folder ze zdjęciami jest pusty!")
else:
    print("BŁĄD: Nie widzę plików. Sprawdź czy nazwy folderów w 'base_dir' są dokładnie takie jak na dysku.")
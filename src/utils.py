import os
import sys
import subprocess
import torch
import numpy as np
import requests
import zipfile

def download_spacy_model():
    """Sprawdza i pobiera model spacy."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        print("Pobieranie modelu Spacy en_core_web_sm...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def ensure_glove_exists(cache_dir='.vector_cache'):
    """Pobiera GloVe, jeśli nie istnieje."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    glove_path = os.path.join(cache_dir, 'glove.6B.300d.txt')
    
    if not os.path.exists(glove_path):
        print("Nie znaleziono GloVe. Pobieranie (862 MB)... to potrwa chwilę.")
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join(cache_dir, "glove.6B.zip")
        
        # Pobieranie
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        # Rozpakowywanie
        print("Rozpakowywanie GloVe...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract('glove.6B.300d.txt', cache_dir)
        
        # Sprzątanie
        os.remove(zip_path)
        print("GloVe gotowe.")
    else:
        print("GloVe znalezione lokalnie.")

def load_glove_embeddings(vocab, embedding_dim=300, cache_dir='.vector_cache'):
    """Ładuje wektory do macierzy."""
    ensure_glove_exists(cache_dir)
    glove_path = os.path.join(cache_dir, 'glove.6B.300d.txt')
    
    print("Ładowanie wektorów do pamięci RAM...")
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    vocab_size = len(vocab)
    embedding_matrix = torch.zeros((vocab_size, embedding_dim))
    hits = 0
    
    for word, idx in vocab.stoi.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = torch.from_numpy(embedding_vector)
            hits += 1
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim)

    print(f"Dopasowano {hits} słów z GloVe (Słownik: {vocab_size}).")
    return embedding_matrix
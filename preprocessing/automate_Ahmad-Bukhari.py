import pandas as pd
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib

def clean_text(text, stopwords, stemmer):
    text = str(text)
    # 1. Lowercase
    text = text.lower()
    # 2. Ganti 'yg' dengan 'yang'
    text = text.replace('yg', 'yang')
    # 3. Hapus URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # 4. Hapus mention & hashtag
    text = re.sub(r'@\w+|#\w+', '', text)
    # 5. Hapus angka
    text = re.sub(r'\d+', '', text)
    # 6. Hapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 7. Hapus whitespace berlebih
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # 8. Tokenisasi
    tokens = text.split()
    # 9. Hapus stopwords
    tokens = [word for word in tokens if word not in stopwords]
    # 10. Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def run_preprocessing(file_path):
    
    print(f"1. Memuat dataset dari: {file_path}")
    df = pd.read_csv(file_path)

    columns_to_drop = ['topik', 'keyword', 'url', 'gambar', 'text_length', 'word_count']
    # Hanya drop kolom yang benar-benar ada di DF
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_drop_columns = df.drop(columns=existing_cols_to_drop)
    
    print("3. Menghapus baris dengan missing value (NA)...")
    df_clean_missing_value = df_drop_columns.dropna()
    
    print("4. Menghapus data duplikat...")
    df_clean_duplicate = df_clean_missing_value.drop_duplicates().copy()
    
    print("5. Menyiapkan Stopword Remover dan Stemmer dari Sastrawi...")
    stop_factory = StopWordRemoverFactory()
    stopwords = set(stop_factory.get_stop_words())
    
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    
    print("6. Memulai proses pembersihan teks (lowercase, hapus url/mention, stopword, stemming)...")
    print("   Catatan: Proses ini mungkin memakan waktu beberapa menit, mohon ditunggu...")
    df_clean_duplicate['clean_text'] = df_clean_duplicate['tweet'].apply(lambda x: clean_text(x, stopwords, stemmer))
    
    print("\n[+] Preprocessing selesai dengan suskse!")
    return df_clean_duplicate

if __name__ == "__main__":
    dataset_path = "../datasetUMPOHoax.csv"
    
    try:
        df_ready = run_preprocessing(dataset_path)
        print("\nPratinjau Hasil Preprocessing:")
        print(df_ready[['tweet', 'clean_text', 'label']].head())
        
    except FileNotFoundError:
        print(f"ERROR: Dataset tidak ditemukan di jalur: {dataset_path}")
        print("Silakan periksa kembali path file CSV yang Anda gunakan.")
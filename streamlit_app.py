import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import re

# Membaca dataset
data = pd.read_excel('JudulBerita.xlsx')

# Preprocessing data
def preprocess_text(text):
  text = text.lower()  # Normalisasi teks menjadi huruf kecil
  text = re.sub(r'[^a-zA-Z\s]', '', text)  # Menghapus karakter non-alfabet
  return text

data['judul_berita'] = data['judul_berita'].apply(preprocess_text)

# Memisahkan fitur dan label
X = data['judul_berita']
y = data['kategori']

# Membagi dataset menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorisasi teks menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Mengembangkan model dengan Linear SVM
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Evaluasi model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Fungsi untuk prediksi kategori
def predict_category(new_text):
    new_text_processed = preprocess_text(new_text)
    new_text_tfidf = vectorizer.transform([new_text_processed])
    return model.predict(new_text_tfidf)[0]

# Aplikasi Streamlit
st.title("Prediksi Kategori Judul Berita")
st.write("Masukkan judul berita untuk memprediksi kategorinya.")

# Input teks dari pengguna
user_input = st.text_input("Masukkan judul berita:")

if user_input:
    predicted_category = predict_category(user_input)
    st.write(f"Kategori untuk judul berita **'{user_input}'** adalah **{predicted_category}**")

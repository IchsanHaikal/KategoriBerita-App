import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords dan tokenizer
nltk.download('stopwords')
nltk.download('punkt')

# Membaca dataset
data = pd.read_excel('JudulBerita.xlsx')

# Preprocessing data
def preprocess_text(text):
    # Periksa apakah teks adalah Tidak Ada dan kembalikan string kosong jika demikian
    if text is None:
        return ''
    # Normalisasi teks menjadi huruf kecil
    text = text.lower()
    
    # Menghapus karakter non-alfabet
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Menghapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Menggabungkan kembali token menjadi teks
    text = ' '.join(filtered_tokens)
    return text # Kembalikan teks yang telah diproses

# Terapkan preprocessing ke dataset
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

# Mengembangkan model dengan Linear SVC
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Evaluasi model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
classification_report_text = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_text)

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
    st.write(f"**Kategori untuk judul berita** '{user_input}' **adalah** {predicted_category}")

# Menampilkan evaluasi model
st.write("### Evaluasi Model")
st.write(f"**Accuracy:** {accuracy}")
st.write(f"**Classification Report:**\n {classification_report_text}")

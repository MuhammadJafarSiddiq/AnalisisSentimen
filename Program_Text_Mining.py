#========================================================================================================================================#
#                                                                  Library                                                               #
#========================================================================================================================================#

import pandas as pd  # Mengimpor library pandas untuk manipulasi data
import re  # Mengimpor library re untuk ekspresi reguler
import matplotlib.pyplot as plt  # Mengimpor matplotlib untuk membuat grafik
from textblob import TextBlob  # Mengimpor TextBlob untuk analisis sentimen
from nltk.corpus import stopwords  # Mengimpor stopwords dari NLTK
from nltk.tokenize import word_tokenize  # Mengimpor tokenisasi kata dari NLTK
from nltk.stem import WordNetLemmatizer  # Mengimpor lemmatizer dari NLTK
from sklearn.feature_extraction.text import TfidfVectorizer  # Mengimpor TfidfVectorizer untuk menghitung TF-IDF
from wordcloud import WordCloud  # Mengimpor WordCloud
import nltk  # Mengimpor NLTK untuk mengunduh stopwords dan punkt
from sklearn.feature_extraction.text import CountVectorizer  # Mengimpor CountVectorizer untuk Bag of Words
import seaborn as sns
from sklearn.cluster import KMeans  # Mengimpor KMeans dari sklearn
from sklearn.decomposition import PCA

#========================================================================================================================================#
#                                                             Text Processing                                                            #
#========================================================================================================================================#

# Load data dari Excel
file_path = 'iphone.xlsx'  # Ganti dengan path file Excel Dataset Anda
data = pd.read_excel(file_path)  # Membaca data dari file Excel ke dalam DataFrame

# Tentukan nama kolom dalam variabel, Pastikan nama kolom ada
column_name = 'reviewDescription' # Ganti dengan nama kolom yang mau di eksekusi
if column_name not in data.columns:
    raise ValueError(f"Kolom '{column_name}' tidak ditemukan dalam data.")  # Menangkap kesalahan jika kolom tidak ada

# Hapus nilai non-string dan ubah menjadi string kosong
data[column_name] = data[column_name].apply(lambda x: x if isinstance(x, str) else '')

# Case folding: Mengubah teks menjadi huruf kecil
data['lowercase'] = data[column_name].str.lower()  # Mengubah teks menjadi huruf kecil

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Hapus URL dari teks
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  
    # Hapus mention (@) dan hashtag (#)
    text = re.sub(r'\@\w+|\#', '', text)  
    # Hapus angka dari teks
    text = re.sub(r'\d+', '', text)  
    # Hapus tanda baca
    text = re.sub(r'[^\w\s]', '', text)  
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi ekstra
    return text  # Mengembalikan teks yang telah dibersihkan

# Bersihkan teks
data['cleaned'] = data['lowercase'].apply(clean_text)  # Menerapkan fungsi clean_text ke kolom yang sudah di-case fold

# Tokenisasi: Memecah teks menjadi kata-kata
data['tokenized'] = data['cleaned'].apply(word_tokenize)  # Menerapkan tokenisasi

# Menghapus stopwords
stop_words = set(stopwords.words('english'))  # Mengambil stopwords bahasa Inggris
data['stopwords'] = data['tokenized'].apply(lambda tokens: [word for word in tokens if word not in stop_words])  # Menghapus stopwords

# Lemmatization: Mengubah kata menjadi bentuk dasar
lemmatizer = WordNetLemmatizer()  # Inisialisasi lemmatizer
data['lemmatized'] = data['stopwords'].apply(lambda words: [lemmatizer.lemmatize(word) for word in words])  # Menerapkan lemmatization

# Mengubah daftar kata yang telah di-lemmatize menjadi string
data['final'] = data['lemmatized'].apply(lambda words: ' '.join(words))  # Menggabungkan kembali kata-kata menjadi string

# Fungsi untuk menghapus kata duplikat
def remove_duplicate_words(text):
    words = text.split()  # Memecah string menjadi kata
    unique_words = list(dict.fromkeys(words))  # Menghapus duplikat dengan menjaga urutan
    return ' '.join(unique_words)  # Menggabungkan kembali kata yang unik menjadi string

# Menerapkan penghapusan duplikat setelah lemmatization
data['final_no_duplicates'] = data['final'].apply(remove_duplicate_words)

#========================================================================================================================================#
#                                                            Sentiment Analysis                                                          #
#========================================================================================================================================#

# Fungsi untuk analisis sentimen
def analyze_sentiment(text):
    analysis = TextBlob(text)  # Menganalisis teks menggunakan TextBlob
    if analysis.sentiment.polarity > 0:  # Jika polaritas positif
        return 'Positif'  # Mengembalikan label 'Positif'
    elif analysis.sentiment.polarity < 0:  # Jika polaritas negatif
        return 'Negatif'  # Mengembalikan label 'Negatif'
    else:
        return 'Netral'  # Mengembalikan label 'Netral' jika polaritas = 0

# Analisis sentimen
data['sentiment'] = data['final_no_duplicates'].apply(analyze_sentiment)  # Menerapkan analisis sentimen ke kolom yang sudah dibersihkan

# Hitung jumlah setiap kategori sentimen
sentiment_counts = data['sentiment'].value_counts()  # Menghitung jumlah setiap kategori sentimen

# Hitung total ulasan
total_reviews = sentiment_counts.sum()  # Total jumlah ulasan dari semua kategori

# Hitung jumlah setiap kategori sentimen
sentiment_counts = data['sentiment'].value_counts()  # Menghitung jumlah setiap kategori sentimen

# Hitung total ulasan
total_reviews = sentiment_counts.sum()  # Total jumlah ulasan dari semua kategori

# Hitung persentase untuk setiap kategori sentimen
sentiment_percentages = (sentiment_counts / total_reviews) * 100  # Menghitung persentase

# Tampilkan jumlah positif, negatif, dan netral di terminal
print("===================================================================================")
print("Jumlah Sentimen:")
print("----------------")
print(f"Positif: {sentiment_counts.get('Positif', 0)} ({sentiment_percentages.get('Positif', 0):.2f}%)")  # Menampilkan jumlah dan persentase sentimen positif
print(f"Negatif: {sentiment_counts.get('Negatif', 0)} ({sentiment_percentages.get('Negatif', 0):.2f}%)")  # Menampilkan jumlah dan persentase sentimen negatif
print(f"Netral: {sentiment_counts.get('Netral', 0)} ({sentiment_percentages.get('Netral', 0):.2f}%)")  # Menampilkan jumlah dan persentase sentimen netral
print("----------------")
print(f"Total Deskripsi: {total_reviews}")  # Menampilkan total deskripsi
# Menampilkan total persentase
total_percentage = sentiment_percentages.sum()
print(f"Total Persentase: {total_percentage:.2f}%")  # Menampilkan total persentase
print("===================================================================================")

# Buat diagram batang
plt.figure(figsize=(8, 6))  # Menentukan ukuran grafik
bars = plt.bar(sentiment_counts.index, sentiment_percentages.values, color=['#66c2a5', '#fc8d62', '#8da0cb'])  # Membuat diagram batang
plt.title('Analisis Sentimen')  # Judul grafik
plt.xlabel('Sentimen')  # Label sumbu X
plt.ylabel('Persentase (%)')  # Label sumbu Y diubah menjadi Persentase
plt.xticks(rotation=45)  # Memutar label sumbu X untuk keterbacaan
plt.ylim(0, 100)  # Mengatur batas sumbu Y dari 0 hingga 100
plt.grid(axis='y')  # Menambahkan grid pada sumbu Y

# Menambahkan persentase di atas batang
for i, bar in enumerate(bars):
    percentage = sentiment_percentages.iloc[i]  # Menggunakan iloc untuk akses berdasarkan posisi
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
            f"{percentage:.2f}%", ha='center', va='bottom', fontsize=10)  # Menampilkan persentase di atas batang

plt.show()  # Menampilkan grafik

#========================================================================================================================================#
#                                                                 TF-IDF                                                                 #
#========================================================================================================================================#

# Inisialisasi TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Hitung TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(data['final_no_duplicates'])

# Dapatkan fitur nama (kata)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Buat DataFrame untuk hasil TF-IDF dengan dokumen di baris dan kata di kolom
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Transpose DataFrame TF-IDF agar kata-kata menjadi baris dan dokumen menjadi kolom
tfidf_dft = tfidf_df.T

# Menghitung total nilai TF-IDF untuk setiap kata
tfidf_sum = tfidf_df.sum(axis=0)  # Menjumlahkan kolom TF-IDF
tfidf_top_words = tfidf_sum.nlargest(10)  # Mengambil 10 kata teratas dengan nilai TF-IDF terbesar

# Menampilkan 10 kata teratas dengan penjelasan
print("===================================================================================")
print("Kata-kata yang Penting dan Relevan Berdasarkan Hasil TF-IDF:")
print("-------------------------------------------------------------")
for i, (word, value) in enumerate(zip(tfidf_top_words.index, tfidf_top_words.values), start=1):
    print(f"{i}. {word}: {value:.4f}") # Menampilkan nomor, kata, dan nilainya
print("-------------------------------------------------------------")
print("Menunjukkan relevansi dan signifikansi kata dalam konteks tertentu, berguna untuk mengekstrak kata-kata yang paling penting dalam analisis teks.")
print("===================================================================================")

#========================================================================================================================================#
#                                                            Visualisasi TF-IDF                                                          #
#========================================================================================================================================#

# Visualisasi hasil TF-IDF menggunakan diagram batang
plt.figure(figsize=(10, 6))
tfidf_top_words.plot(kind='barh', color='skyblue')

# Menambahkan judul dan label sumbu
plt.title('10 Kata Teratas Berdasarkan Nilai TF-IDF', fontsize=16)
plt.xlabel('Nilai TF-IDF', fontsize=14)
plt.ylabel('Kata', fontsize=14)
plt.gca().invert_yaxis()  # Membalik sumbu y agar kata teratas muncul di atas

# Menambahkan nilai TF-IDF di sebelah kanan batang
for index, value in enumerate(tfidf_top_words.values):
    plt.text(value + 0.05, index, f'{value:.4f}', va='center')  # Menambahkan offset pada value untuk memberikan jarak dari batang

# Menampilkan diagram
plt.show()

#========================================================================================================================================#
#                                                                Bag of Words                                                            #
#========================================================================================================================================#

# Inisialisasi CountVectorizer untuk menghitung frekuensi kata (Bag of Words)
bow_vectorizer = CountVectorizer()

# Hitung BoW (Bag of Words) dari teks yang sudah dibersihkan dan dihapus duplikatnya
bow_matrix = bow_vectorizer.fit_transform(data['final_no_duplicates'])

# Dapatkan fitur nama (kata)
bow_feature_names = bow_vectorizer.get_feature_names_out()

# Buat DataFrame untuk hasil BoW dengan dokumen di baris dan kata di kolom
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_feature_names)

# Transpose DataFrame BoW agar kata-kata menjadi baris dan dokumen menjadi kolom
bow_dft = bow_df.T

# Menghitung total kemunculan setiap kata
bow_sum = bow_df.sum(axis=0)  # Menjumlahkan kolom BoW
bow_top_words = bow_sum.nlargest(10)  # Mengambil 10 kata teratas dengan kemunculan terbanyak

# Menampilkan 10 kata teratas dengan penjelasan
print("===================================================================================")
print("Kata-kata yang Paling Sering Muncul Berdasarkan Hasil Bag of Words (BoW):")
print("-----------------------------------------------------------------------")
for i, (word, value) in enumerate(zip(bow_top_words.index, bow_top_words.values), start=1):
    print(f"{i}. {word}: {value}") # Menampilkan nomor, kata, dan jumlah kemunculan
print("-----------------------------------------------------------------------")
print("Menunjukkan frekuensi kemunculan kata dalam dokumen, berguna untuk memahami kata-kata yang paling umum.")
print("===================================================================================")

#========================================================================================================================================#
#                                                      Visualisasi Bag of Words                                                          #
#========================================================================================================================================#

# Membuat DataFrame untuk kata-kata paling sering muncul
bow_top_words_df = pd.DataFrame({
    'word': bow_top_words.index,
    'frequency': bow_top_words.values
})

# Membuat diagram batang menggunakan seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='frequency', y='word', data=bow_top_words_df, palette='viridis', hue='word', dodge=False, legend=False)

# Menambahkan judul dan label sumbu
plt.title('10 Kata Teratas Berdasarkan Frekuensi Bag of Words', fontsize=16)
plt.xlabel('Frekuensi', fontsize=14)
plt.ylabel('Kata', fontsize=14)

# Menambahkan angka frekuensi di sebelah kanan batang
for index, value in enumerate(bow_top_words_df['frequency']):
    plt.text(value + 3, index, f'{value}', va='center')  # Menambahkan offset pada value untuk memberikan jarak dari batang

# Menampilkan diagram
plt.show()

#========================================================================================================================================#
#                                                                Save to                                                                 #
#========================================================================================================================================#

# Simpan hasil analisis sentimen ke file Excel
output_sentiment_file_path = 'hasil_analisis_sentimen.xlsx'  # Menentukan path untuk file output analisis sentimen
# Reset index dan ubah dimulai dari 1
data_reset = data.reset_index(drop=True)  # Reset index tanpa menyimpan index lama
data_reset.index += 1  # Tambahkan 1 ke setiap index agar mulai dari 1
data_reset.to_excel(output_sentiment_file_path, index=True)  # Menyimpan DataFrame analisis sentimen ke file Excel

# Simpan hasil TF-IDF ke file Excel
output_tfidf_file_path = 'hasil_tfidf.csv'  # Menentukan path untuk file output TF-IDF
tfidf_dft.columns = range(1, len(tfidf_dft.columns) + 1)  # Indeks kolom mulai dari 1
tfidf_dft.to_csv(output_tfidf_file_path, index=True)  # Menyimpan DataFrame TF-IDF ke file Excel

# Simpan hasil Bag of Words (BoW) ke file Excel
output_bow_file_path = 'hasil_bow.csv'  # Menentukan path untuk file output BoW
bow_dft.columns = range(1, len(bow_dft.columns) + 1)  # Indeks kolom mulai dari 1
bow_dft.to_csv(output_bow_file_path, index=True)  # Menyimpan DataFrame BoW ke file CSV

#========================================================================================================================================#
#                                                         Word Cloud TF-IDF                                                              #
#========================================================================================================================================#

def plot_wordcloud(tfidf_matrix, feature_names):
    # Menghitung total nilai TF-IDF untuk setiap kata
    tfidf_sum = tfidf_matrix.sum(axis=0).A1  # Menjumlahkan kolom TF-IDF
    word_tfidf = dict(zip(feature_names, tfidf_sum))  # Menggabungkan kata dan nilai TF-IDF dalam dictionary
    
    # Membuat Word Cloud dengan semua kata
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_tfidf)

    # Tampilkan Word Cloud
    plt.figure(figsize=(10, 5))  # Menentukan ukuran grafik
    plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan Word Cloud
    plt.axis('off')  # Menghilangkan sumbu
    plt.title('Word Cloud dari Semua Kata TF-IDF')  # Judul grafik
    plt.show()  # Menampilkan grafik

# Panggil fungsi untuk memplot Word Cloud dengan semua kata
plot_wordcloud(tfidf_matrix, feature_names)

#========================================================================================================================================#
#                                                       Word Cloud Bag of Words                                                          #
#========================================================================================================================================#

def plot_wordcloud_bow(bow_matrix, feature_names):
    # Menghitung total frekuensi untuk setiap kata dari BoW
    bow_sum = bow_matrix.sum(axis=0).A1  # Menjumlahkan kolom BoW
    word_bow = dict(zip(feature_names, bow_sum))  # Menggabungkan kata dan frekuensi dalam dictionary
    
    # Membuat Word Cloud dengan semua kata
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_bow)

    # Tampilkan Word Cloud
    plt.figure(figsize=(10, 5))  # Menentukan ukuran grafik
    plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan Word Cloud
    plt.axis('off')  # Menghilangkan sumbu
    plt.title('Word Cloud dari Bag of Words')  # Judul grafik
    plt.show()  # Menampilkan grafik

# Panggil fungsi untuk memplot Word Cloud dengan BoW
plot_wordcloud_bow(bow_matrix, bow_feature_names)

#========================================================================================================================================#
#                                                             K-Means Clustering                                                         #
#========================================================================================================================================#

# Menggunakan KMeans untuk clustering teks berdasarkan TF-IDF
num_clusters = 3  # Tentukan jumlah cluster yang diinginkan
kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # Inisialisasi KMeans
kmeans.fit(tfidf_matrix)  # Melatih model KMeans

# Menambahkan label cluster ke DataFrame
data['cluster'] = kmeans.labels_  # Menambahkan label cluster ke DataFrame

# Menampilkan jumlah ulasan dalam setiap cluster
cluster_counts = data['cluster'].value_counts()
print("===================================================================================")
print("Jumlah Ulasan per Cluster:")
print(cluster_counts)
print("===================================================================================")

# Menampilkan centroid dari cluster
centroids = kmeans.cluster_centers_
print("Centroid dari masing-masing cluster:")
for i in range(num_clusters):
    print(f"Cluster {i}: {centroids[i]}")  # Menampilkan centroid dari setiap cluster

# Visualisasi hasil clustering dengan menampilkan angka di atas batang
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data, x='cluster', hue='cluster', palette='viridis', legend=False)
plt.title('Jumlah Ulasan per Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Jumlah Ulasan', fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y')

# Menambahkan angka di atas batang
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.show()

#========================================================================================================================================#
#                                                               PCA Visualization                                                        #
#========================================================================================================================================#

# Menggunakan PCA untuk mengurangi dimensi dan memvisualisasikan hasil clustering
pca = PCA(n_components=2)  # Mengurangi ke 2 dimensi
reduced_data = pca.fit_transform(tfidf_matrix.toarray())  # Melatih PCA pada data TF-IDF

# Membuat DataFrame untuk hasil PCA
pca_df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = data['cluster']  # Menambahkan label cluster ke DataFrame PCA

# Visualisasi hasil PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='cluster', palette='viridis', legend=False, alpha=0.6)
plt.title('PCA Visualization of Clusters', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=14)
plt.ylabel('PCA Component 2', fontsize=14)
plt.grid()
plt.show()

#========================================================================================================================================#
#                                                    Menyimpan Hasil Clustering ke Excel                                               #
#========================================================================================================================================#

# Menyimpan DataFrame ke file Excel
output_file = 'hasil_clustering.xlsx'  # Nama file output
data.to_excel(output_file, index=False)  # Menyimpan DataFrame tanpa index

#========================================================================================================================================#
#                                                                  Selesai                                                               #
#========================================================================================================================================#
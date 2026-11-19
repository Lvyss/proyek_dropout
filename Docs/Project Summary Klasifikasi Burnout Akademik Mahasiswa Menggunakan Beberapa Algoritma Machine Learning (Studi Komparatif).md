# Project Summary: Klasifikasi Dropout dan Keberhasilan Akademik Mahasiswa Menggunakan Beberapa Algoritma Machine Learning (Studi Komparatif)

Link dataset: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

## 1. Overview Proyek

**Judul:** Klasifikasi Dropout dan Keberhasilan Akademik Mahasiswa Menggunakan Beberapa Algoritma Machine Learning: Studi Komparatif

**Tipe Proyek:** Tugas Akhir Mata Kuliah Data Science / Data Mining
**Bentuk Deliverable:** Web Application (dashboard interaktif) + Artikel/Laporan
**Bahasa Pemrograman:** Python
**Framework Web:** Streamlit
**Kerangka Kerja:** CRISP-DM (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment)

**Ringkasan singkat:**
Proyek ini membangun dashboard interaktif berbasis web. Pengguna meng-upload dataset mahasiswa dalam format CSV, lalu melakukan eksplorasi data (EDA), preprocessing, menjalankan 7 algoritma klasifikasi dari scikit-learn (ditambah baseline Zero-R), dan membandingkan performanya melalui tabel evaluasi, confusion matrix, ROC-AUC, serta hasil validasi silang. Dataset tidak tersedia sejak awal di aplikasi, melainkan di-upload oleh pengguna.

---

## 2. Latar Belakang

Dropout mahasiswa adalah masalah yang berdampak pada mahasiswa, keluarga, dan institusi pendidikan. Faktor penyebabnya beragam: latar belakang pendaftaran, kondisi sosial-ekonomi, status pembayaran, hingga performa akademik pada semester awal. Karena faktornya banyak dan saling berkaitan, pendekatan machine learning relevan untuk mengidentifikasi pola dan memprediksi status akhir mahasiswa secara lebih objektif daripada hanya mengandalkan observasi manual.

Penelitian ini memakai dataset dari Instituto Politécnico de Portalegre (Portugal) yang dibuat untuk mengidentifikasi mahasiswa berisiko sejak dini. Targetnya tiga kelas: _Dropout_, _Enrolled_, dan _Graduate_. Kelas dalam data ini tidak seimbang, sehingga pemilihan metrik evaluasi menjadi bagian penting dari studi.

## 3. Urgensi / Kenapa Penting Diteliti

- Deteksi dini mahasiswa berisiko dropout memungkinkan intervensi akademik dan finansial lebih cepat.
- Institusi dapat memanfaatkan model prediktif untuk merancang kebijakan pendampingan yang lebih tepat sasaran.
- Studi komparatif memberi gambaran algoritma mana yang paling sesuai untuk data tabular pendidikan yang bersifat multi-kelas dan tidak seimbang.
- Karena tidak ada satu model yang terbaik untuk semua kasus (_No Free Lunch_), perbandingan empiris tetap diperlukan.

## 4. Rumusan Masalah

1. Bagaimana karakteristik data mahasiswa (latar belakang pendaftaran, sosial-ekonomi, dan performa akademik) berkaitan dengan status akhir mahasiswa (Dropout, Enrolled, Graduate)?
2. Algoritma machine learning manakah yang memberikan performa klasifikasi terbaik dalam memprediksi status akhir mahasiswa?
3. Bagaimana perbandingan hasil evaluasi (accuracy, precision, recall, F1-score, ROC-AUC, serta waktu komputasi) dari ketujuh algoritma, dan bagaimana pengaruh metode validasi serta hyperparameter tuning terhadap hasilnya?

## 5. Tujuan Penelitian

1. Menganalisis karakteristik dan pola data mahasiswa melalui exploratory data analysis (EDA).
2. Mengimplementasikan dan membandingkan performa 7 algoritma klasifikasi pada kasus multi-kelas dengan data tidak seimbang.
3. Mengidentifikasi algoritma dengan performa terbaik berdasarkan metrik evaluasi, validasi silang, dan kriteria pemilihan model (akurasi, kecepatan, interpretabilitas).

**Kriteria keberhasilan (CRISP-DM):** model terbaik adalah model dengan macro-F1 tertinggi pada data uji, hasilnya stabil pada validasi silang (standar deviasi kecil), dan selisih skor train-test tidak besar (tidak overfit). Baseline Zero-R menjadi batas minimal yang harus dilampaui.

---

## 6. Dataset

**Nama Dataset:** Predict Students' Dropout and Academic Success
**Sumber:** UCI Machine Learning Repository (ID 697)
**Pembuat:** Realinho, Vieira Martins, Machado, & Baptista (2021), Instituto Politécnico de Portalegre
**Lisensi:** CC BY 4.0 (wajib disitasi)
**Format:** CSV, pemisah titik koma (`;`), encoding UTF-8 dengan BOM

**Hasil pengecekan file:**

| Aspek              | Temuan                                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| Jumlah baris       | 4.424 (memenuhi syarat minimal 1.000 baris)                                                          |
| Jumlah kolom       | 37 (36 fitur + 1 kolom `Target`)                                                                     |
| Missing value      | Tidak ada                                                                                            |
| Baris duplikat     | Tidak ada                                                                                            |
| Distribusi target  | Graduate 2.209 (49,9%), Dropout 1.421 (32,1%), Enrolled 794 (17,9%)                                  |
| Catatan nama kolom | Kolom `Daytime/evening attendance` mengandung karakter tab di akhir nama; perlu di-strip saat dibaca |

**Alasan pemilihan dataset:**

- Data nyata, bukan sintetis, dan memiliki paper pendukung sehingga hasilnya dapat dipertanggungjawabkan.
- Ukuran cukup untuk melatih dan menguji 7 algoritma tanpa beban komputasi berlebihan.
- Target multi-kelas (3 kelas) dengan ketidakseimbangan yang nyata, sehingga relevan untuk membahas accuracy paradox, macro-F1, dan penanganan imbalance.
- Kombinasi fitur numerik, biner, dan nominal berkode angka membuat preprocessing menjadi bermakna, bukan formalitas.

**Struktur kolom (gambaran umum):**

| Kategori                   | Contoh Kolom                                                                                                     |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Demografis                 | Marital status, Gender, Age at enrollment, Nacionality, International                                            |
| Jalur pendaftaran          | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Admission grade |
| Latar belakang keluarga    | Mother's/Father's qualification, Mother's/Father's occupation                                                    |
| Sosial-ekonomi & finansial | Displaced, Debtor, Tuition fees up to date, Scholarship holder, Educational special needs                        |
| Akademik semester 1 dan 2  | Curricular units (credited, enrolled, evaluations, approved, grade, without evaluations)                         |
| Makroekonomi               | Unemployment rate, Inflation rate, GDP                                                                           |
| Target                     | Target (Dropout / Enrolled / Graduate)                                                                           |

**Karakteristik tipe data (penting untuk preprocessing):**

- Delapan kolom bersifat biner (mis. Gender, Debtor, Scholarship holder).
- Banyak kolom nominal tersimpan sebagai angka kode (Course, Application mode, Marital status, Previous qualification, Nacionality, kualifikasi dan pekerjaan orang tua). Angka ini adalah kode kategori, bukan nilai berurutan, sehingga harus di-encode sebagai kategorikal (One-Hot Encoding).
- Application order bersifat ordinal (0 sampai 9).
- Kolom nilai dan tingkat ekonomi bersifat numerik kontinu.

---

## 7. Metodologi

### 7.1 Algoritma yang Digunakan (7 Metode Scikit-Learn + 1 Baseline)

| No  | Algoritma                                            | Kegunaan                                             |
| --- | ---------------------------------------------------- | ---------------------------------------------------- |
| 0   | Zero-R (`DummyClassifier`, strategi kelas mayoritas) | Baseline minimal                                     |
| 1   | Logistic Regression                                  | Baseline model linear/probabilistik                  |
| 2   | Decision Tree                                        | Model berbasis aturan, mudah diinterpretasi          |
| 3   | Random Forest                                        | Ensemble dari banyak decision tree                   |
| 4   | K-Nearest Neighbors (KNN)                            | Klasifikasi berbasis kedekatan data (_lazy learner_) |
| 5   | Support Vector Machine (SVM)                         | Klasifikasi berbasis margin/hyperplane               |
| 6   | Naive Bayes                                          | Klasifikasi probabilistik                            |
| 7   | Gradient Boosting                                    | Ensemble boosting untuk performa lebih tinggi        |

_(Catatan: daftar bisa disesuaikan dengan ketentuan dosen. Karena targetnya 3 kelas, penanganan multi-kelas mengikuti strategi bawaan scikit-learn, misalnya One-vs-One pada SVM dan One-vs-Rest/multinomial pada Logistic Regression.)_

### 7.2 Tahapan Preprocessing

1. **Parsing & pembersihan nama kolom:** deteksi delimiter otomatis, encoding `utf-8-sig`, strip spasi/tab pada nama kolom.
2. **Data quality check:** missing value, duplikat, outlier.
3. **Penanganan missing value:** drop / isi mean / isi median / isi modus (dataset ini tidak memiliki missing value, namun fitur dibuat umum untuk dataset lain).
4. **Penanganan outlier (opsional):** none / winsorizing berbasis IQR pada kolom numerik kontinu.
5. **Encoding:** One-Hot Encoding untuk kolom nominal; Label Encoding untuk kolom target.
6. **Scaling:** none / MinMaxScaler / StandardScaler (wajib untuk KNN, SVM, dan Logistic Regression).
7. **Seleksi fitur (opsional):** pilih/hapus kolom manual, atau SelectKBest / PCA untuk mengurangi dimensi.
8. **Penanganan imbalance (opsional):** `class_weight="balanced"` atau oversampling (mis. SMOTE), hanya diterapkan pada data latih.
9. **Splitting data:** train-test split terstratifikasi dengan rasio yang dapat diatur (default 80:20).

> Semua transformasi (scaler, encoder, imputer, resampling) hanya di-_fit_ pada data latih lalu diterapkan ke data uji, memakai `Pipeline` + `ColumnTransformer`, untuk mencegah data leakage.

### 7.3 Skema Validasi dan Tuning

- **Holdout** (train-test split terstratifikasi, default 80:20).
- **Stratified k-fold Cross Validation** (default k = 10) untuk estimasi performa yang lebih stabil; ditampilkan sebagai rata-rata ± standar deviasi.
- **Hyperparameter tuning** dengan Grid Search atau Random Search (dengan grid kecil agar tetap cepat di web), dievaluasi memakai cross validation.

### 7.4 Metrik Evaluasi

- Accuracy (disertai pembahasan _accuracy paradox_ karena kelas tidak seimbang)
- Precision, Recall, F1-Score (macro dan weighted average)
- Confusion Matrix 3×3
- ROC-AUC multi-kelas (One-vs-Rest, macro)
- Skor train vs skor test (untuk mendeteksi overfitting/underfitting)
- Waktu training dan waktu prediksi
- Interpretabilitas model (kualitatif)

---

## 8. Gambaran Aplikasi Web

### 8.1 Alur Upload (Dataset Tidak Tersedia dari Awal)

Aplikasi tidak membawa dataset bawaan. Sebelum ada file yang di-upload, semua tab hanya menampilkan petunjuk upload. Setelah file di-upload, aplikasi:

1. Membaca file dengan pengaturan yang tahan terhadap variasi format CSV:

```python
import pandas as pd

def load_csv(file):
    df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df
```

_(Sudah diuji pada file dataset ini: terbaca 4.424 baris × 37 kolom. Tanpa `sep=None`, file ini terbaca sebagai 1 kolom karena pemisahnya titik koma.)_

2. Memvalidasi file: tidak kosong, memiliki minimal 2 kolom, dan kolom target dapat dipilih.
3. **Pengguna memilih kolom target** lewat dropdown (default: kolom bernama `Target` atau kolom terakhir). Aplikasi memeriksa target memiliki minimal 2 kelas dan setiap kelas cukup besar untuk split terstratifikasi/CV.
4. **Penentuan tipe kolom:** aplikasi menebak tipe (numerik / nominal / biner), lalu pengguna dapat mengoreksinya lewat multiselect. Langkah ini diperlukan karena kolom nominal berkode angka (misalnya `Course`) otomatis terbaca sebagai numerik.
5. Menyimpan data di `st.cache_data` agar tidak dibaca ulang setiap interaksi.

### 8.2 Arsitektur Umum

```
[User] → [Upload CSV] → [Validasi + Pilih Target + Tipe Kolom] → [Dashboard Streamlit]
                                                                         │
        ┌───────────┬───────────────┬──────────────────┬────────────────┘
        │           │               │                  │
    Overview       EDA        Preprocessing         Evaluasi        Parameter
                                                                    (Sidebar)
```

### 8.3 Menu Utama

**1. Overview**

- Menampilkan data mentah setelah upload.
- Info dasar: jumlah baris, jumlah kolom, tipe data per kolom, ukuran file.

**2. EDA (Exploratory Data Analysis)**

- Statistik deskriptif (mean, median, std, kuartil).
- Distribusi kelas target (bar chart) dan persentase tiap kelas.
- Distribusi per kolom (histogram, boxplot).
- Scatter plot antar dua fitur numerik (diwarnai berdasarkan kelas target).
- Correlation heatmap antar fitur numerik.
- Cek missing value, baris duplikat, dan outlier.
- Opsional: uji normalitas / Q-Q plot untuk fitur numerik.

**3. Preprocessing**

- Pilihan metode lewat dropdown/checkbox: missing value, duplikat, outlier, encoding, scaling, seleksi fitur, penanganan imbalance.
- Output: tabel data hasil preprocessing beserta ringkasan perubahan (jumlah kolom sebelum dan sesudah).

**4. Evaluasi**

- Pengguna memilih algoritma (multi-select) dan metode validasi (holdout atau stratified k-fold CV).
- Tabel hasil: accuracy, macro precision/recall/F1, weighted F1, ROC-AUC, skor train vs test, waktu training/prediksi, per algoritma.
- Confusion matrix per model dan kurva ROC.
- Visualisasi perbandingan antar model (bar chart).
- Opsi hyperparameter tuning (Grid/Random Search) dengan hasil sebelum vs sesudah tuning.

### 8.4 Sidebar: Parameter

Pengaturan menyesuaikan tab aktif:

- **Preprocessing:** strategi missing value, metode scaling, penanganan outlier, pemilihan kolom/fitur.
- **Evaluasi:** rasio train-test, jumlah fold CV, `random_state`, hyperparameter dasar, misalnya `C` (Logistic Regression/SVM), `max_depth` (Decision Tree), `n_estimators` (Random Forest/Gradient Boosting), `k` (KNN).

---

## 9. Tech Stack

| Komponen                        | Tools                       |
| ------------------------------- | --------------------------- |
| Bahasa                          | Python                      |
| Web Framework                   | Streamlit                   |
| Data Processing                 | Pandas, NumPy               |
| Machine Learning                | Scikit-Learn                |
| Penanganan Imbalance (opsional) | imbalanced-learn            |
| Visualisasi                     | Matplotlib, Seaborn, Plotly |
| Deployment (opsional)           | Streamlit Community Cloud   |

---

## 10. Keterkaitan dengan Materi Kuliah

| Materi                         | Penerapan di Proyek                                                                                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| PPT 1 – Introduction           | Klasifikasi sebagai supervised learning; kerangka CRISP-DM; Nine Laws (Data Preparation Law, NFL-DM)                                                |
| PPT 2 – Data Understanding     | Tab Overview dan EDA: statistik deskriptif, histogram, boxplot, scatter plot, korelasi, tipe atribut, verifikasi kualitas data                      |
| PPT 3 – Preprocessing          | Tab Preprocessing: missing value, duplikat, outlier, encoding, normalisasi (min-max/z-score), stratified sampling, balancing, feature selection/PCA |
| PPT 4 – Classification         | 7 algoritma + baseline Zero-R; penanganan multi-kelas (One-vs-One/One-vs-Rest); kriteria pemilihan model (akurasi, kecepatan, interpretabilitas)    |
| PPT 5 – Evaluating Classifiers | Holdout, stratified k-fold CV, confusion matrix, accuracy paradox, precision/recall/F1, ROC-AUC, hyperparameter tuning, overfitting/underfitting    |

---

## 11. Rencana Pengerjaan (Roadmap)

| Tahap | Deskripsi                                                                      | Estimasi   |
| ----- | ------------------------------------------------------------------------------ | ---------- |
| 1     | Finalisasi dataset dan topik; konfirmasi ke dosen                              | Minggu 1   |
| 2     | Penyusunan proposal (latar belakang, rumusan masalah, tujuan, kriteria sukses) | Minggu 1-2 |
| 3     | EDA awal di notebook (distribusi kelas, tipe kolom, korelasi)                  | Minggu 2   |
| 4     | Modul upload + validasi + tab Overview & EDA                                   | Minggu 3   |
| 5     | Tab Preprocessing (pipeline tanpa data leakage)                                | Minggu 3-4 |
| 6     | Implementasi 7 algoritma + Zero-R, holdout dan CV, tab Evaluasi                | Minggu 4-5 |
| 7     | Hyperparameter tuning, ROC-AUC, analisis overfitting                           | Minggu 5-6 |
| 8     | Testing (termasuk uji upload dengan variasi CSV), debugging, penyempurnaan UI  | Minggu 6   |
| 9     | Penyusunan artikel/laporan akhir dan persiapan presentasi                      | Minggu 6-7 |

---

## 12. Catatan Tambahan dan Keterbatasan

- Perlu klarifikasi ke dosen: apakah ke-7 algoritma sudah ditentukan atau bebas dipilih, serta apakah proyek individu atau kelompok.
- Pastikan dataset final tidak duplikat dengan kelompok/individu lain di kelas (via spreadsheet pengumpulan).
- Dataset ini memuat nilai akademik semester 1 dan 2, sehingga prediksi baru bisa dilakukan setelah dua semester berjalan. Untuk menguji skenario deteksi lebih dini, bandingkan hasil memakai semua fitur dengan hasil tanpa kolom `Curricular units`.
- Kelas _Enrolled_ adalah kelas minoritas dan umumnya paling sulit diprediksi; bahas recall per kelas di bagian hasil.
- Aplikasi bersifat umum untuk dataset klasifikasi tabular, tetapi seluruh analisis dan kesimpulan artikel difokuskan pada dataset UCI ini.

## Sitasi

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). _Predict Students' Dropout and Academic Success_ [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. _Trends and Applications in Information Systems and Technologies._

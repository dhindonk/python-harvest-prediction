# Sistem Prediksi Hasil Panen

Sistem ini dirancang untuk memprediksi hasil panen pertanian menggunakan teknologi machine learning dengan model LSTM (Long Short-Term Memory). Sistem ini terdiri dari dua komponen utama: modul training model dan aplikasi web untuk prediksi.

## Daftar Isi

1. [Proses Training Model](#proses-training-model)
2. [Aplikasi Prediksi](#aplikasi-prediksi)
3. [Cara Menjalankan Project](#cara-menjalankan-project)

---

## Proses Training Model

Proses training model menggunakan pendekatan KDD (Knowledge Discovery in Databases) yang dimodifikasi sesuai dengan kebutuhan prediksi hasil panen. File training utama adalah [`train_model_v9.py`](train_model_v9.py:1).

### 1. Seleksi Data (Data Selection)

**Lokasi Kode:** [`train_model_v9.py`](train_model_v9.py:86-105)

Pada tahap ini, sistem memuat dan membersihkan data mentah dari file CSV atau Excel:

```python
def load_and_clean(self, file_path):
    print("\n[1/6] Loading and cleaning data...")
    try:
        if file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        print(f"   ✓ Loaded {len(df)} rows")
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
```

**Penjelasan Rinci:**
Sistem membaca data pertanian dari file CSV atau Excel yang berisi informasi seperti suhu, curah hujan, dosis pupuk, dan hasil panen. Data ini kemudian dibersihkan dengan mengubah nama kolom menjadi format yang konsisten (misalnya "Suhu C" menjadi "suhu") untuk memudahkan proses selanjutnya.

### 2. Preprocessing Data

**Lokasi Kode:** [`train_model_v9.py`](train_model_v9.py:107-131)

Tahap preprocessing meliputi pembuatan indeks waktu dan agregasi data bulanan:

```python
def create_datetime_index(self, df):
    print("\n[2/6] Creating datetime index...")
    try:
        if 'tahun' in df.columns and 'bulan' in df.columns:
            if 'tanggal' not in df.columns:
                df['tanggal'] = 1
            df['tanggal_dt'] = pd.to_datetime(
                df[['tahun', 'bulan', 'tanggal']].rename(
                    columns={'tahun': 'year', 'bulan': 'month', 'tanggal': 'day'}
                ), errors='coerce'
            )
```

**Penjelasan Rinci:**
Data harian diubah menjadi format waktu yang terstruktur, kemudian diagregasi menjadi data bulanan. Ini penting karena prediksi panen lebih relevan pada skala bulanan daripada harian. Misalnya, data suhu harian dirata-ratakan menjadi suhu bulanan, dan hasil panen dijumlahkan per bulan.

### 3. Transformasi Data (Data Transformation)

**Lokasi Kode:** [`train_model_v9.py`](train_model_v9.py:153-180) dan [`train_model_v9.py`](train_model_v9.py:182-223)

Pada tahap ini, sistem melakukan dekomposisi STL dan rekayasa fitur:

```python
def stl_decomposition(self, df, target_col='hasil_panen_kg', seasonal=13):
    print("\n[4/6] Performing STL decomposition...")
    try:
        if len(df) < 2 * 12:
            print(f"   ⚠ Data terlalu pendek ({len(df)} bulan) untuk STL. Menggunakan dummy.")
            self.trend_component = np.zeros(len(df))
            self.seasonal_component = np.zeros(len(df))
            return None
        seasonal_period = 13
        if len(df) < 30: seasonal_period = 7
```

**Penjelasan Rinci:**
Sistem memecah data hasil panen menjadi tiga komponen utama:
- **Tren**: Pola jangka panjang (apakah hasil panen cenderung meningkat atau menurun)
- **Musiman**: Pola berulang setiap tahun (misalnya panen tinggi di musim hujan)
- **Sisa**: Variabilitas acak yang tidak dapat dijelaskan oleh tren atau musim

Kemudian, sistem membuat fitur-fitur baru seperti:
- **Lag features**: Data hasil panen dari bulan-bulan sebelumnya
- **Rolling statistics**: Rata-rata dan standar deviasi hasil panen dalam periode tertentu
- **Fitur interaksi**: Kombinasi antara curah hujan dan suhu
- **Fitur waktu**: Representasi bulan dan kuartal dalam format matematis

### 4. Pemodelan Data (Data Mining)

**Lokasi Kode:** [`train_model_v9.py`](train_model_v9.py:263-300)

Sistem menggunakan arsitektur LSTM khusus untuk prediksi multi-output:

```python
class EnhancedMultiOutputLSTM:
    def __init__(self, sequence_length, n_features, output_steps=12):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.output_steps = output_steps
        self.model = None
        self.history = None
    
    def build_model(self, lstm_units=64, dropout_rate=0.3, l1_reg=1e-5, l2_reg=1e-4):
        print(f"\n[MODEL] Building Enhanced Multi-Output LSTM...")
        try:
            model = Sequential([
                Input(shape=(self.sequence_length, self.n_features), name='input_layer'),
                LSTM(units=lstm_units, activation='tanh', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), return_sequences=True, name='lstm_1'),
                Dropout(dropout_rate),
                LSTM(units=lstm_units // 2, activation='tanh', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), return_sequences=False, name='lstm_2'),
                Dropout(dropout_rate),
                RepeatVector(self.output_steps),
                LSTM(units=lstm_units // 2, activation='tanh', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), recurrent_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), return_sequences=True, name='lstm_decoder'),
                Dropout(dropout_rate),
                TimeDistributed(Dense(1, activation='relu', kernel_constraint=NonNegativeConstraint(), name='time_distributed'))
            ])
```

**Penjelasan Rinci:**
Model LSTM (Long Short-Term Memory) adalah jenis jaringan saraf tiruan yang dirancang khusus untuk mengenali pola dalam data berurutan seperti data waktu. Model ini bekerja seperti otak yang dapat "mengingat" pola dari data masa lalu untuk memprediksi masa depan.

Arsitektur model ini memiliki:
- **2 lapisan LSTM encoder**: Mempelajari pola dari data historis
- **1 lapisan LSTM decoder**: Menghasilkan prediksi untuk 12 bulan ke depan
- **Dropout**: Mencegah model "hafal" data (overfitting)
- **Regularisasi**: Memastikan model bekerja baik pada data baru
- **Kendala non-negatif**: Memastikan prediksi hasil panen tidak pernah negatif

### 5. Evaluasi Model

**Lokasi Kode:** [`train_model_v9.py`](train_model_v9.py:51-69)

Sistem menggunakan fungsi loss khusus yang disesuaikan untuk data pertanian:

```python
class BalancedVarianceMSE(keras.losses.Loss):
    def __init__(self, variance_weight=0.05, seasonal_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.variance_weight = variance_weight
        self.seasonal_weight = seasonal_weight
    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        pred_variance = tf.math.reduce_variance(y_pred)
        target_variance = tf.math.reduce_variance(y_true)
        variance_diff = tf.abs(pred_variance - target_variance)
        variance_loss = variance_diff / (target_variance + 1e-8)
        monthly_diffs = tf.abs(y_pred[:, 1:, :] - y_pred[:, :-1, :])
        seasonal_loss = tf.reduce_mean(monthly_diffs)
        total_loss = mse + self.variance_weight * variance_loss + self.seasonal_weight * seasonal_loss
        return total_loss
```

**Penjelasan Rinci:**
Sistem tidak hanya mengukur seberapa dekat prediksi dengan nilai asli (MSE), tetapi juga:
- **Variance matching**: Memastikan variasi prediksi mirip dengan variasi data asli
- **Seasonal consistency**: Memastikan pola musiman dalam prediksi masuk akal
- **Smooth transitions**: Memastikan perubahan antar bulan tidak terlalu ekstrem

### 6. Penyimpanan Model

**Lokasi Kode:** [`train_model_v9.py`](train_model_v9.py:326-337) dan [`train_model_v9.py`](train_model_v9.py:342-367)

Model dan parameter preprocessing disimpan secara terpisah untuk portabilitas:

```python
def save_model_weights(self, filepath_prefix='harvest_lstm_enhanced'):
    print(f"\n[SAVING MODEL WEIGHTS]")
    try:
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_path = f"models/{filepath_prefix}_weights_{timestamp}.weights.h5"
        self.model.save_weights(weights_path)
        print(f"   ✓ Model weights saved: {weights_path}")
        return weights_path, timestamp
    except Exception as e:
        print(f"   ⚠ Save weights failed: {e}")
        return None, None

def save_processor_json(processor, sequence_length, timestamp, filepath_prefix='harvest_processor'):
    processor_data = {
        'scaler_features_center': processor.scaler_features.center_.tolist(),
        'scaler_features_scale': processor.scaler_features.scale_.tolist(),
        'scaler_target_min': processor.scaler_target.min_.tolist(),
        'scaler_target_scale': processor.scaler_target.scale_.tolist(),
        'feature_columns': processor.feature_columns,
        'sequence_length': sequence_length
    }
```

**Penjelasan Rinci:**
Setelah training selesai, sistem menyimpan:
- **Bobot model**: "Pengetahuan" yang dipelajari oleh model LSTM
- **Parameter preprocessing**: Informasi tentang cara data diolah (skala, fitur yang digunakan, dll.)

Penyimpanan terpisah ini memungkinkan model dijalankan di lingkungan yang berbeda tanpa harus melakukan training ulang.

---

## Aplikasi Prediksi

Aplikasi web dibangun menggunakan Flask untuk menyediakan antarmuka pengguna yang intuitif. File utama aplikasi adalah [`app.py`](app.py:1).

### 1. Struktur Aplikasi

**Lokasi Kode:** [`app.py`](app.py:249-258)

Aplikasi menggunakan arsitektur Flask dengan model dan processor yang dimuat secara global:

```python
app = Flask(__name__)
app.secret_key = 'harvest_prediction_secret_key_v-final'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variabel Global
model = None
processor = None
model_loaded = False
sequence_length_from_model = 12 # Default
```

**Penjelasan Rinci:**
Aplikasi web dibangun dengan Flask, sebuah framework Python yang ringan untuk membuat aplikasi web. Aplikasi ini menyediakan halaman web di mana pengguna dapat mengunggah data pertanian dan mendapatkan hasil prediksi dalam bentuk visual yang mudah dipahami.

### 2. Pemuatan Model

**Lokasi Kode:** [`app.py`](app.py:260-332)

Sistem memuat model dan parameter preprocessing saat aplikasi dimulai:

```python
def load_model_and_processor():
    global model, processor, model_loaded, sequence_length_from_model
    
    try:
        # 1. Cari file bobot (.weights.h5) 
        weights_files = glob.glob('models/harvest_lstm_enhanced_weights_*.weights.h5')
        if not weights_files:
            print("[ERROR] Tidak ada file model weights (.weights.h5) di folder /models")
            return
        latest_weights_path = max(weights_files, key=os.path.getmtime)

        # 2. Cari file parameter processor (.json) 
        processor_files = glob.glob('models/harvest_processor_params_*.json')
        if not processor_files:
            print("[ERROR] Tidak ada file processor params (.json) di folder /models")
            return
        latest_processor_path = max(processor_files, key=os.path.getmtime)
```

**Penjelasan Rinci:**
Saat aplikasi dimulai, sistem secara otomatis mencari dan memuat model yang telah dilatih sebelumnya. Sistem menggunakan file terbaru yang tersedia di folder models. Proses ini memastikan aplikasi siap digunakan tanpa perlu training ulang.

### 3. Proses Prediksi

**Lokasi Kode:** [`app.py`](app.py:386-498)

Sistem memproses data yang diunggah pengguna dan menghasilkan prediksi:

```python
@app.route('/predict_year', methods=['POST'])
def predict_year():
    global model, processor, model_loaded, sequence_length_from_model
    if not model_loaded or processor is None or model is None:
        flash('Model belum siap. Harap tunggu atau cek log server.', 'error')
        return redirect(url_for('index'))
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
```

**Penjelasan Rinci:**
Ketika pengguna mengunggah file data, sistem melakukan langkah-langkah berikut:
1. **Validasi file**: Memastikan file yang diunggah adalah CSV atau Excel
2. **Preprocessing**: Membersihkan dan memproses data dengan cara yang sama seperti saat training
3. **Prediksi**: Menggunakan model LSTM untuk memprediksi hasil panen 12 bulan ke depan
4. **Analisis**: Menghasilkan kesimpulan dan rekomendasi berdasarkan hasil prediksi
5. **Visualisasi**: Menampilkan hasil dalam bentuk grafik dan tabel yang mudah dipahami

### 4. Antarmuka Pengguna

**Lokasi Kode:** [`templates/index_simple.html`](templates/index_simple.html:1) dan [`templates/results_yearly.html`](templates/results_yearly.html:1)

Aplikasi menggunakan template HTML modern dengan Tailwind CSS untuk tampilan yang responsif dan menarik:

```html
<div class="text-center mb-12">
  <h1 class="text-4xl md:text-5xl font-bold text-primary mb-4">
    <i class="fas fa-seedling mr-3"></i> Sistem Prediksi Hasil Panen
  </h1>
  <p class="text-lg text-gray-600 max-w-2xl mx-auto">
    Upload file CSV data pertanian untuk mendapatkan prediksi hasil panen
    menggunakan model LSTM
  </p>
</div>
```

**Penjelasan Rinci:**
Antarmuka pengguna dirancang agar mudah digunakan bahkan untuk orang yang tidak memiliki latar belakang teknis. Pengguna hanya perlu:
1. Mengunduh template data (opsional)
2. Mengunggah file data pertanian
3. Menekan tombol "Prediksi 1 Tahun"
4. Melihat hasil prediksi dalam bentuk grafik dan tabel
5. Mengunduh hasil prediksi dalam format PDF

---

## Cara Menjalankan Project

Berikut adalah langkah-langkah lengkap untuk menjalankan sistem prediksi hasil panen:

### 1. Persyaratan Sistem

Pastikan sistem Anda memiliki:
- Python 3.8 atau lebih tinggi
- Git (untuk cloning repository) 

### 2. Setup Environment

```bash
# Clone repository
git clone https://github.com/dhindonk/python-harvest-prediction.git
cd python-harvest-prediction

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Untuk Windows:
venv\Scripts\activate
# Untuk Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install semua library yang dibutuhkan
pip install -r requirements.txt
```

**Penjelasan Rinci:**
File [`requirements.txt`](requirements.txt:1) berisi daftar semua library Python yang dibutuhkan sistem ini, seperti:
- **pandas**: Untuk memproses data tabular
- **numpy**: Untuk operasi matematika
- **tensorflow**: Untuk machine learning dan LSTM
- **flask**: Untuk aplikasi web
- **scikit-learn**: Untuk preprocessing data
- **matplotlib/seaborn**: Untuk visualisasi
- **statsmodels**: Untuk analisis time series

### 4. Training Model (Optional)

```bash
# Jika model belum ada, jalankan training
python train_model_v9.py
```

**Penjelasan Rinci:**
Langkah ini bersifat opsional karena model yang sudah dilatih seharusnya sudah tersedia di folder `models`. Proses training mungkin memakan waktu beberapa menit hingga beberapa jam tergantung pada ukuran data dan spesifikasi komputer.

### 5. Menjalankan Aplikasi

```bash
# Jalankan aplikasi web
python app.py
```
---

## Struktur Project

```
harvest-prediction/
├── app.py                          # Aplikasi Flask utama
├── train_model_v9.py              # Script training model
├── requirements.txt                # Daftar dependencies
├── README.md                      # Dokumentasi project
├── data/                          # Folder untuk data training
├── models/                        # Folder untuk model yang sudah dilatih
├── uploads/                       # Folder untuk file yang diunggah
├── static/                        # File statis (CSS, JS, gambar)
│   └── js/
│       └── pdf-generator.js       # Script untuk generate PDF
└── templates/                     # Template HTML
    ├── index_simple.html          # Halaman utama
    ├── results_yearly.html        # Halaman hasil prediksi
    ├── articles.html             # Halaman artikel
    └── article_detail.html       # Halaman detail artikel
```

--- 
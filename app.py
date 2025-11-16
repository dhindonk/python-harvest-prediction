# -*- coding: utf-8 -*-
"""
Flask App for Harvest Prediction - FINAL (Standalone & JSON Portable)
Description:
  - STANDALONE: Semua kelas (Processor, Model, Loss, Constraint)
    diduplikasi dari skrip training.
  - PORTABEL (JSON): Memuat HANYA bobot (.weights.h5) dan parameter 
    scaler dari .json untuk menghindari SEMUA error versi (Keras & NumPy).
"""

import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import json # <-- Impor JSON
import warnings
import io
import glob
import re
import sys

# --- IMPOR SEMUA DEPENDENSI LOKAL ---
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import STL
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed, Layer
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# =============================================================================
# --- KELAS-KELAS DIDUPLIKASI DARI TRAINING_PIPELINE ---
# =============================================================================
class NonNegativeConstraint(Constraint):
    def __call__(self, w): return w * tf.cast(tf.greater_equal(w, 0.), w.dtype)
    def get_config(self): return {}

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
    def get_config(self):
        config = super().get_config()
        config.update({'variance_weight': self.variance_weight, 'seasonal_weight': self.seasonal_weight})
        return config

class EnhancedDataProcessor:
    def __init__(self, aggregation_freq='MS'):
        self.aggregation_freq = aggregation_freq
        self.scaler_features = RobustScaler()
        self.scaler_target = MinMaxScaler()
        self.decomposition = None
        self.feature_columns = None
        self.seasonal_component = None
        self.trend_component = None
        self.target_min = None
        self.target_max = None
    
    def load_and_clean(self, file_path):
        print("\n[APP-PROC: 1/6] Loading and cleaning data...")
        try:
            if file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            print(f"   ✓ Loaded {len(df)} rows")
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            rename_map = {
                'suhu_c': 'suhu', 'curah_hujan_mm': 'curah_hujan',
                'dosis_pupuk_(kg)': 'dosis_pupuk_kg',
                'pelepah_terkena_penyakit_(kg)': 'penyakit',
                'pelepah_terkena_luka_goresan_(kg)': 'luka_goresan',
            }
            df.rename(columns=rename_map, inplace=True)
            return df
        except Exception as e:
            print(f"   ⚠ Loading failed: {e}")
            raise
    
    def create_datetime_index(self, df):
        print("\n[APP-PROC: 2/6] Creating datetime index...")
        try:
            if 'tahun' in df.columns and 'bulan' in df.columns:
                if 'tanggal' not in df.columns:
                    df['tanggal'] = 1
                df['tanggal_dt'] = pd.to_datetime(
                    df[['tahun', 'bulan', 'tanggal']].rename(
                        columns={'tahun': 'year', 'bulan': 'month', 'tanggal': 'day'}
                    ), errors='coerce'
                )
            else:
                date_col = next((col for col in df.columns if 'tanggal' in col or 'date' in col), None)
                if date_col:
                     df['tanggal_dt'] = pd.to_datetime(df[date_col], errors='coerce')
                else:
                    raise ValueError("Tidak ditemukan kolom 'tahun'/'bulan' atau 'tanggal'/'date'")
            df.dropna(subset=['tanggal_dt'], inplace=True)
            df.set_index('tanggal_dt', inplace=True)
            df.sort_index(inplace=True)
            print(f"   ✓ Date range: {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            print(f"   ⚠ Datetime creation failed: {e}")
            raise
    
    def aggregate_to_monthly(self, df):
        print("\n[APP-PROC: 3/6] Aggregating to monthly data...")
        try:
            agg_rules = {
                'hasil_panen_kg': 'sum', 'curah_hujan': 'sum', 'suhu': 'mean',
                'kelembapan': 'mean', 'dosis_pupuk_kg': 'sum', 'umur_tanaman': 'mean',
                'luas_lahan': 'first', 'penyakit': 'sum', 'luka_goresan': 'sum'
            }
            valid_agg = {k: v for k, v in agg_rules.items() if k in df.columns}
            if 'hasil_panen_kg' not in valid_agg:
                raise ValueError("Kolom 'hasil_panen_kg' tidak ditemukan.")
            df_monthly = df.resample('MS').agg(valid_agg)
            df_monthly['luas_lahan'] = df_monthly['luas_lahan'].ffill().bfill()
            df_monthly = df_monthly.fillna(0)
            print(f"   ✓ Created {len(df_monthly)} monthly records")
            return df_monthly
        except Exception as e:
            print(f"   ⚠ Monthly aggregation failed: {e}")
            raise
    
    def stl_decomposition(self, df, target_col='hasil_panen_kg', seasonal=13):
        print("\n[APP-PROC: 4/6] Performing STL decomposition...")
        try:
            if len(df) < 2 * 12:
                print(f"   ⚠ Data terlalu pendek ({len(df)} bulan) untuk STL. Menggunakan dummy.")
                self.trend_component = np.zeros(len(df))
                self.seasonal_component = np.zeros(len(df))
                return None
            seasonal_period = 13
            if len(df) < 30: seasonal_period = 7
            from statsmodels import __version__ as sm_version
            if str(sm_version) >= '0.14.0':
                stl = STL(df[target_col], seasonal=seasonal_period, robust=True)
            else:
                stl = STL(df[target_col], seasonal=seasonal_period)
            result = stl.fit()
            self.decomposition = result
            self.trend_component = result.trend.values
            self.seasonal_component = result.seasonal.values
            print(f"   ✓ STL decomposition completed (Seasonal={seasonal_period})")
            return result
        except Exception as e:
            print(f"   ⚠ STL decomposition failed: {e}")
            self.trend_component = np.zeros(len(df))
            self.seasonal_component = np.zeros(len(df))
            return None
    
    def engineer_features(self, df, target_col='hasil_panen_kg'):
        print("\n[APP-PROC: 5/6] Engineering features...")
        try:
            df_feat = df.copy()
            if self.trend_component is not None and len(self.trend_component) == len(df_feat):
                df_feat['trend_component'] = self.trend_component
                df_feat['seasonal_component'] = self.seasonal_component
            else:
                df_feat['trend_component'] = 0
                df_feat['seasonal_component'] = 0
            for lag in [1, 2, 3, 6, 12]:
                if lag < len(df_feat):
                    df_feat[f'hasil_panen_kg_lag_{lag}'] = df_feat[target_col].shift(lag)
            for window in [2, 3, 6]:
                if window < len(df_feat):
                    df_feat[f'hasil_panen_kg_rolling_mean_{window}'] = df_feat[target_col].shift(1).rolling(window=window, min_periods=1).mean()
                    df_feat[f'hasil_panen_kg_rolling_std_{window}'] = df_feat[target_col].shift(1).rolling(window=window, min_periods=1).std()
            df_feat = df_feat.fillna(method='bfill').fillna(method='ffill').fillna(0)
            if 'curah_hujan' not in df_feat.columns: df_feat['curah_hujan'] = 0
            if 'suhu' not in df_feat.columns: df_feat['suhu'] = 0
            if 'dosis_pupuk_kg' not in df_feat.columns: df_feat['dosis_pupuk_kg'] = 0
            if 'luas_lahan' not in df_feat.columns: df_feat['luas_lahan'] = 0
            if 'penyakit' not in df_feat.columns: df_feat['penyakit'] = 0
            if 'luka_goresan' not in df_feat.columns: df_feat['luka_goresan'] = 0
            df_feat['rainfall_x_temp'] = df_feat['curah_hujan'] * df_feat['suhu']
            df_feat['fertilizer_density'] = df_feat['dosis_pupuk_kg'] / (df_feat['luas_lahan'] + 1e-6)
            df_feat['total_damage'] = df_feat['penyakit'] + df_feat['luka_goresan']
            df_feat['damage_ratio'] = df_feat['total_damage'] / (df_feat[target_col] + 1e-6)
            df_feat['month'] = df_feat.index.month
            df_feat['quarter'] = df_feat.index.quarter
            df_feat['year'] = df_feat.index.year
            df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
            df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
            df_feat['quarter_sin'] = np.sin(2 * np.pi * df_feat['quarter'] / 4)
            df_feat['quarter_cos'] = np.cos(2 * np.pi * df_feat['quarter'] / 4)
            initial_len = len(df_feat)
            df_feat.dropna(inplace=True)
            print(f"   ✓ Created features, dropped {initial_len - len(df_feat)} rows")
            return df_feat
        except Exception as e:
            print(f"   ⚠ Feature engineering failed: {e}")
            raise

class EnhancedMultiOutputLSTM:
    def __init__(self, sequence_length, n_features, output_steps=12):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.output_steps = output_steps
        self.model = None
    
    def build_model(self, lstm_units=64, dropout_rate=0.3, l1_reg=1e-5, l2_reg=1e-4):
        print(f"\n[APP-MODEL] Building local model architecture...")
        try:
            try:
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, clipnorm=1.0)
            except AttributeError:
                print("   [INFO] tf.keras.optimizers.legacy.Adam tidak ditemukan. Menggunakan tf.keras.optimizers.Adam.")
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

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
            self.model = model
            print(f"   ✓ Local model built successfully. Parameters: {model.count_params():,}")
            return model
        except Exception as e:
            print(f"   ⚠ Local model building failed: {e}")
            raise
# =============================================================================
# --- AKHIR DARI KELAS-KELAS YANG DIDUPLIKASI ---
# =============================================================================

app = Flask(__name__)
app.secret_key = 'harvest_prediction_secret_key_v-final'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variabel Global
model = None
processor = None
model_loaded = False
sequence_length_from_model = 12 # Default

def load_model_and_processor():
    global model, processor, model_loaded, sequence_length_from_model
    
    print(f"[ENV] TensorFlow (Lokal): {tf.__version__}")
    print(f"[ENV] NumPy (Lokal): {np.__version__}")

    try:
        # 1. Cari file bobot (.weights.h5) terbaru
        weights_files = glob.glob('models/harvest_lstm_enhanced_weights_*.weights.h5')
        if not weights_files:
            print("[ERROR] Tidak ada file model weights (.weights.h5) di folder /models")
            print("         Harap jalankan 'training_lokal.py' terlebih dahulu.")
            return
        latest_weights_path = max(weights_files, key=os.path.getmtime)

        # 2. Cari file parameter processor (.json) terbaru
        processor_files = glob.glob('models/harvest_processor_params_*.json') # <-- Cari .json
        if not processor_files:
            print("[ERROR] Tidak ada file processor params (.json) di folder /models")
            print("         Pastikan Anda menjalankan 'training_lokal.py' terlebih dahulu.")
            return
        latest_processor_path = max(processor_files, key=os.path.getmtime)
        
        print(f"[LOAD] Memuat bobot dari: {latest_weights_path}")
        print(f"[LOAD] Memuat parameter dari: {latest_processor_path}")

        # 3. Muat PARAMETER Processor (sebagai DICT dari JSON)
        with open(latest_processor_path, 'r') as f:
            processor_params = json.load(f) # <-- Muat dari JSON
        
        # 4. "Hidrasi Ulang" (Re-hydrate) Processor lokal
        processor = EnhancedDataProcessor() # Buat instance baru
        
        processor.scaler_features.center_ = np.array(processor_params['scaler_features_center'])
        processor.scaler_features.scale_ = np.array(processor_params['scaler_features_scale'])
        processor.scaler_target.min_ = np.array(processor_params['scaler_target_min'])
        processor.scaler_target.scale_ = np.array(processor_params['scaler_target_scale'])
        
        processor.feature_columns = processor_params['feature_columns']
        sequence_length_from_model = processor_params['sequence_length']
        
        print("[LOAD] Parameter processor (JSON) berhasil di-hydrate secara lokal.")
        
        # 5. Bangun Ulang Arsitektur Model
        n_features = len(processor.feature_columns)
        output_steps = 12 # Asumsi 12 bulan
        
        model_wrapper = EnhancedMultiOutputLSTM(sequence_length_from_model, n_features, output_steps)
        model = model_wrapper.build_model()
        
        # 6. Muat Bobot (Weights)
        model.load_weights(latest_weights_path)
        print("[LOAD] Bobot (weights) berhasil dimuat ke arsitektur lokal.")

        # 7. Kompilasi Ulang Model (Wajib setelah load weights)
        try:
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        except AttributeError:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        loss = BalancedVarianceMSE()
        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        print("[LOAD] Model berhasil di-recompile.")
            
        model_loaded = True
        print(f"\n✓✓✓ Model (weights) & Processor (JSON params) berhasil dimuat. Aplikasi siap. ✓✓✓")

    except Exception as e:
        print(f"✗✗✗ Gagal total memuat model: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False

# --- Rute Flask ---
@app.route('/')
def index():
    # Asumsi Anda memiliki 'index_simple.html' di folder 'templates'
    try:
        return render_template('index_simple.html', model_loaded=model_loaded)
    except Exception as e:
        print(f"Warning: Gagal me-render template 'index_simple.html'. {e}")
        return f"<h1>Error</h1><p>Template 'index_simple.html' tidak ditemukan di folder 'templates'.</p><p>Model loaded: {model_loaded}</p>"


@app.route('/articles')
def articles():
    try:
        return render_template('articles.html')
    except Exception:
        return "Halaman Artikel (template 'articles.html' tidak ditemukan)"

@app.route('/article/<article_id>')
def article_detail(article_id):
    try:
        articles_data = {
            'panduan-budidaya-lidah-buaya': {
                'id': 'panduan-budidaya-lidah-buaya',
                'title': 'Panduan Lengkap Budidaya Lidah Buaya untuk Pemula',
                'category': 'Budidaya',
                'read_time': '5',
                'publish_date': '15 Oktober 2024',
                'excerpt': 'Pelajari teknik budidaya lidah buaya yang efektif mulai dari pemilihan bibit hingga panen, termasuk perawatan optimal untuk hasil maksimal.',
                'gradient_color': 'from-green-400 to-green-600',
                'icon': 'fas fa-leaf',
                'badge_color': 'bg-green-100 text-green-800',
                'content': '''
                <h2>Pengenalan Tanaman Lidah Buaya</h2>
                <p>Lidah buaya (Aloe vera) adalah tanaman sukulen yang telah lama dikenal karena manfaatnya untuk kesehatan dan kecantikan. Tanaman ini relatif mudah dibudidayakan dan memiliki potensi ekonomi yang tinggi. Di Indonesia, lidah buaya dapat tumbuh dengan baik di berbagai kondisi iklim, terutama di daerah dengan intensitas cahaya matahari yang cukup.</p>
                
                <p>Budidaya lidah buaya menjadi pilihan yang menarik bagi petani karena beberapa alasan:</p>
                <ul>
                    <li>Perawatan relatif mudah dan tidak memerlukan perhatian khusus</li>
                    <li>Tahan terhadap kondisi kekeringan</li>
                    <li>Pasar yang luas untuk produk olahan lidah buaya</li>
                    <li>Dapat dipanen beberapa kali dalam setahun</li>
                    <li>Nilai jual yang stabil dan cenderung meningkat</li>
                </ul>

                <h2>Pemilihan Lokasi dan Persiapan Lahan</h2>
                <p>Lokasi merupakan faktor krusial dalam keberhasilan budidaya lidah buaya. Pilihlah lokasi dengan karakteristik berikut:</p>
                
                <h3>Syarat Iklim</h3>
                <ul>
                    <li>Suhu optimal: 20-30°C</li>
                    <li>Kelembapan relatif: 40-60%</li>
                    <li>Intensitas cahaya matahari: 6-8 jam per hari</li>
                    <li>Curah hujan: 500-800 mm per tahun</li>
                </ul>

                <h3>Persiapan Lahan</h3>
                <p>Lahan yang akan digunakan untuk budidaya lidah buaya perlu dipersiapkan dengan baik:</p>
                <ol>
                    <li><strong>Bersihkan lahan</strong> dari rumput liar dan sisa tanaman sebelumnya</li>
                    <li><strong>Cangkul tanah</strong> hingga kedalaman 30-40 cm</li>
                    <li><strong>Tambahkan pupuk kandang</strong> sebanyak 10-15 ton per hektar</li>
                    <li><strong>Buat bedengan</strong> dengan tinggi 20-30 cm dan lebar 1-1,2 meter</li>
                    <li><strong>Buat saluran drainase</strong> untuk mencegah genangan air</li>
                </ol>

                <h2>Pemilihan Bibit Berkualitas</h2>
                <p>Bibit yang berkualitas adalah fondasi keberhasilan budidaya lidah buaya. Perhatikan kriteria berikut saat memilih bibit:</p>
                
                <h3>Kriteria Bibit Baik</h3>
                <ul>
                    <li>Umur bibit minimal 6-8 bulan</li>
                    <li>Tinggi tanaman 15-20 cm dengan 4-5 daun</li>
                    <li>Daun tebal, berdaging, dan berwarna hijau segar</li>
                    <li>Bebas dari penyakit dan hama</li>
                    <li>Akar yang kuat dan sehat</li>
                </ul>

                <h3>Sumber Bibit</h3>
                <p>Bibit lidah buaya dapat diperoleh melalui:</p>
                <ul>
                    <li><strong>Pemisahan anakan</strong> dari tanaman induk yang sehat</li>
                    <li><strong>Pembelian</strong> dari penangkar resmi</li>
                    <li><strong>Perbanyakan vegetatif</strong> melalui stek daun</li>
                </ul>

                <h2>Teknik Penanaman yang Tepat</h2>
                <p>Penanaman yang tepat akan mempengaruhi pertumbuhan dan hasil panen:</p>
                
                <h3>Jarak Tanam</h3>
                <p>Jarak tanam yang direkomendasikan:</p>
                <ul>
                    <li>Jarak antar baris: 80-100 cm</li>
                    <li>Jarak dalam baris: 60-80 cm</li>
                    <li>Kepadatan: 10.000-15.000 tanaman per hektar</li>
                </ul>

                <h3>Cara Penanaman</h3>
                <ol>
                    <li>Buat lubang tanam dengan kedalaman 15-20 cm</li>
                    <li>Masukkan bibit secara hati-hati</li>
                    <li>Tutup lubang dengan tanah hingga batas leher akar</li>
                    <li>Padatkan tanah di sekitar pangkal batang</li>
                    <li>Siram dengan air secukupnya</li>
                    <li>Buat naungan sementara jika cuaca terlalu panas</li>
                </ol>

                <h2>Perawatan Rutin</h2>
                <p>Perawatan yang baik akan memastikan pertumbuhan optimal dan hasil panen yang maksimal:</p>
                
                <h3>Penyiraman</h3>
                <ul>
                    <li>Frekuensi: 2-3 kali seminggu saat musim kemarau</li>
                    <li>Kurangi penyiraman saat musim hujan</li>
                    <li>Hindari genangan air yang dapat menyebabkan pembusukan akar</li>
                    <li>Siram di pagi hari untuk mengurangi penguapan</li>
                </ul>

                <h3>Pemupukan</h3>
                <p>Jadwal pemupukan yang disarankan:</p>
                <ul>
                    <li><strong>Pupuk dasar</strong>: pupuk kandang saat persiapan lahan</li>
                    <li><strong>Pupuk susulan</strong>: NPK 15-15-15 sebanyak 200-300 kg/ha setiap 3 bulan</li>
                    <li><strong>Pupuk daun</strong>: semprotkan pupuk daun setiap 2 bulan sekali</li>
                </ul>

                <h3>Pengendalian Hama dan Penyakit</h3>
                <p>Hama dan penyakit umum pada lidah buaya:</p>
                <ul>
                    <li><strong>Hama ulat</strong>: gunakan pestisida organik</li>
                    <li><strong>Kutu daun</strong>: semprot dengan air sabun</li>
                    <li><strong>Busuk akar</strong>: kurangi penyiraman dan berikan fungisida</li>
                    <li><strong>Bercak daun</strong>: gunakan fungisida sesuai anjuran</li>
                </ul>

                <h2>Panen dan Pasca Panen</h2>
                <p>Panen yang tepat waktu akan menghasilkan produk berkualitas tinggi:</p>
                
                <h3>Waktu Panen</h3>
                <ul>
                    <li>Umur tanaman siap panen: 8-12 bulan</li>
                    <li>Panen saat daun mencapai panjang 40-60 cm</li>
                    <li>Waktu terbaik: pagi hari setelah embun mengering</li>
                </ul>

                <h3>Teknik Panen</h3>
                <ol>
                    <li>Pilih daun yang matang dan sehat</li>
                    <li>Potong daun di bagian pangkal dengan pisau tajam</li>
                    <li>Sisakan 2-3 daun terlama untuk pertumbuhan lanjutan</li>
                    <li>Sortir dan klasifikasikan hasil panen</li>
                    <li>Bersihkan dari kotoran dan potongan kecil</li>
                </ol>

                <h2>Analisis Ekonomi</h2>
                <p>Berikut estimasi ekonomi budidaya lidah buaya per hektar:</p>
                
                <h3>Biaya Produksi</h3>
                <ul>
                    <li>Persiapan lahan: Rp 5.000.000</li>
                    <li>Pembelian bibit: Rp 15.000.000</li>
                    <li>Pupuk dan pestisida: Rp 8.000.000/tahun</li>
                    <li>Tenaga kerja: Rp 12.000.000/tahun</li>
                    <li>Total biaya tahun pertama: Rp 40.000.000</li>
                </ul>

                <h3>Pendapatan</h3>
                <ul>
                    <li>Hasil panen: 15-20 ton/ha</li>
                    <li>Harga jual: Rp 5.000-8.000/kg</li>
                    <li>Total pendapatan: Rp 75.000.000-160.000.000</li>
                    <li>ROI: 87-300%</li>
                </ul>

                <h2>Kesimpulan</h2>
                <p>Budidaya lidah buaya merupakan peluang bisnis pertanian yang menjanjikan dengan ROI yang tinggi. Dengan mengikuti panduan ini secara disiplin, pemula dapat berhasil membudidayakan lidah buaya dan meraih keuntungan yang optimal. Kunci keberhasilan terletak pada pemilihan bibit berkualitas, perawatan yang konsisten, dan manajemen pasca panen yang baik.</p>
                '''
            },
            'strategi-meningkatkan-panen': {
                'id': 'strategi-meningkatkan-panen',
                'title': '7 Strategi Efektif Meningkatkan Hasil Panen Lidah Buaya',
                'category': 'Optimasi',
                'read_time': '7',
                'publish_date': '12 Oktober 2024',
                'excerpt': 'Temukan strategi terbukti untuk meningkatkan produktivitas panen lidah buaya hingga 40% dengan teknik pertanian modern.',
                'gradient_color': 'from-blue-400 to-blue-600',
                'icon': 'fas fa-chart-line',
                'badge_color': 'bg-blue-100 text-blue-800',
                'content': '''
                <h2>Pendahuluan: Maksimalkan Potensi Lidah Buaya Anda</h2>
                <p>Dalam dunia pertanian modern, meningkatkan hasil panen bukan lagi sekadar keinginan, melainkan keharusan untuk tetap kompetitif. Lidah buaya, dengan segala manfaatnya, memiliki potensi luar biasa jika dibudidayakan dengan teknik yang tepat. Artikel ini akan mengungkap 7 strategi terbukti yang dapat meningkatkan hasil panen lidah buaya hingga 40%.</p>
                
                <p>Sebelum kita masuk ke strategi detail, penting untuk memahami bahwa faktor-faktor seperti varietas, kondisi lingkungan, dan manajemen budidaya saling terkait dan harus dioptimalkan secara holistik.</p>

                <h2>Strategi 1: Seleksi Varietas Unggul dan Adaptif</h2>
                <p>Varietas adalah fondasi keberhasilan budidaya. Memilih varietas yang tepat dapat meningkatkan hasil panen hingga 25%.</p>
                
                <h3>Varietas Rekomendasi</h3>
                <ul>
                    <li><strong>Aloe vera barbadensis</strong>: hasil tinggi, tahan penyakit</li>
                    <li><strong>Aloe vera chinensis</strong>: adaptif, pertumbuhan cepat</li>
                    <li><strong>Aloe vera arborescens</strong>: kualitas gel terbaik</li>
                </ul>

                <h3>Kriteria Seleksi Varietas</h3>
                <ol>
                    <li>Produktivitas tinggi (15-20 ton/ha)</li>
                    <li>Tahan terhadap penyakit umum</li>
                    <li>Adaptif terhadap lokal iklim</li>
                    <li>Kualitas gel yang baik</li>
                    <li>Umur panen relatif singkat (8-10 bulan)</li>
                </ol>

                <h2>Strategi 2: Optimasi Sistem Tanam Jarak Legowo</h2>
                <p>Sistem tanam jarak legowo telah terbukti meningkatkan hasil panen hingga 15% dengan memaksimalkan penggunaan lahan dan intensitas cahaya.</p>
                
                <h3>Konfigurasi Jarak Legowo</h3>
                <ul>
                    <li><strong>Legowo 2:1</strong>: jarak 80cm x 40cm x 120cm</li>
                    <li><strong>Legowo 3:1</strong>: jarak 75cm x 35cm x 100cm</li>
                    <li><strong>Legowo 4:1</strong>: jarak 70cm x 30cm x 90cm</li>
                </ul>

                <h3>Keunggulan Sistem Legowo</h3>
                <ul>
                    <li>Peningkatan populasi tanaman 20-25%</li>
                    <li>Sirkulasi udara lebih baik</li>
                    <li>Pencahayaan lebih merata</li>
                    <li>Mudah dalam pemeliharaan</li>
                </ul>

                <h2>Strategi 3: Manajemen Nutrisi Presisi</h2>
                <p>Pemberian nutrisi yang tepat pada waktu yang tepat dapat meningkatkan hasil panen hingga 20%.</p>
                
                <h3>Analisis Tanah dan Kebutuhan Nutrisi</h3>
                <p>Lakukan analisis tanah setiap 6 bulan untuk menentukan:</p>
                <ul>
                    <li>pH tanah optimal: 6.0-7.0</li>
                    <li>Kandungan organik: >3%</li>
                    <li>NPK rasio ideal: 2:1:2</li>
                </ul>

                <h3>Jadwal Pemupukan Presisi</h3>
                <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
                    <tr style="background-color: #f3f4f6;">
                        <th style="padding: 8px; text-align: left; border: 1px solid #d1d5db;">Fase Pertumbuhan</th>
                        <th style="padding: 8px; text-align: left; border: 1px solid #d1d5db;">Jenis Pupuk</th>
                        <th style="padding: 8px; text-align: left; border: 1px solid #d1d5db;">Dosis (kg/ha)</th>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">Vegetatif (0-4 bulan)</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">NPK 20-10-10</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">200</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">Generatif (4-8 bulan)</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">NPK 15-15-15</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">250</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">Panen (>8 bulan)</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">NPK 10-20-10</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">150</td>
                    </tr>
                </table>

                <h2>Strategi 4: Teknik Irigasi Tetes Efisien</h2>
                <p>Sistem irigasi tetes dapat menghemat air hingga 60% dan meningkatkan hasil panen hingga 12%.</p>
                
                <h3>Keunggulan Irigasi Tetes</h3>
                <ul>
                    <li>Penggunaan air lebih efisien</li>
                    <li>Nutrisi langsung ke zona akar</li>
                    <li>Penurunan pertumbuhan gulma</li>
                    <li>Penurunan kelembaban daun (mengurangi penyakit)</li>
                </ul>

                <h3>Parameter Irigasi Optimal</h3>
                <ul>
                    <li>Frekuensi: 2-3 kali seminggu</li>
                    <li>Durasi: 2-3 jam per penyiraman</li>
                    <li>Debit air: 2-4 liter/tanaman/hari</li>
                    <li>Waktu terbaik: pagi hari (06.00-08.00)</li>
                </ul>

                <h2>Strategi 5: Pengendalian Hama Terpadu (PHT)</h2>
                <p>Pendekatan PHT dapat mengurangi kerugian hasil panen hingga 30% dengan biaya yang lebih efisien.</p>
                
                <h3>Prinsip PHT untuk Lidah Buaya</h3>
                <ol>
                    <li><strong>Monitoring rutin</strong>: deteksi dini hama dan penyakit</li>
                    <li><strong>Batas ekonomi</strong>: intervensi hanya jika kerugian >10%</li>
                    <li><strong>Pengendalian biologis</strong>: musuh alami hama</li>
                    <li><strong>Pestisida selektif</strong>: gunakan sebagai pilihan terakhir</li>
                </ol>

                <h3>Hama Utama dan Pengendaliannya</h3>
                <ul>
                    <li><strong>Ulat grayak</strong>: Bacillus thuringiensis</li>
                    <li><strong>Kutu daun</strong>: predator alami (laba-laba)</li>
                    <li><strong>Tungau</strong>: akarisida selektif</li>
                    <li><strong>Penyakit busuk akar</strong>: Trichoderma sp.</li>
                </ul>

                <h2>Strategi 6: Panen Bertahap dan Pasca Panen Optimal</h2>
                <p>Teknik panen yang tepat dapat meningkatkan kualitas hasil hingga 15% dan memperpanjang umur simpan.</p>
                
                <h3>Teknik Panen Bertahap</h3>
                <ul>
                    <li>Panen 30% daun terlama setiap 3 bulan</li>
                    <li>Sisakan 70% daun untuk fotosintesis</li>
                    <li>Potong di pangkal dengan sudut 45°</li>
                    <li>Gunakan pisau steril untuk mencegah infeksi</li>
                </ul>

                <h3>Teknik Pasca Panen</h3>
                <ol>
                    <li><strong>Sorting</strong>: pisahkan daun berkualitas premium</li>
                    <li><strong>Washing</strong>: bersihkan dengan air mengalir</li>
                    <li><strong>Cooling</strong>: suhu 4-8°C dalam 2 jam</li>
                    <li><strong>Packaging</strong>: bahan bernapas (mesh bags)</li>
                    <li><strong>Storage</strong>: 85-90% kelembapan relatif</li>
                </ol>

                <h2>Strategi 7: Implementasi Teknologi Pertanian 4.0</h2>
                <p>Adopsi teknologi modern dapat meningkatkan efisiensi dan hasil panen hingga 35%.</p>
                
                <h3>IoT untuk Monitoring Pertumbuhan</h3>
                <ul>
                    <li><strong>Sensor kelembaban tanah</strong>: optimalisasi irigasi</li>
                    <li><strong>Sensor suhu dan kelembapan udara</strong>: kontrol mikroklima</li>
                    <li><strong>Sensor nutrisi tanah</strong>: pemupukan presisi</li>
                    <li><strong>Drone untuk monitoring</strong>: deteksi stress tanaman</li>
                </ul>

                <h3>Aplikasi Prediksi Panen</h3>
                <p>Sistem prediksi berbasis AI seperti yang kami kembangkan dapat:</p>
                <ul>
                    <li>Memprediksi hasil panen 3-6 bulan ke depan</li>
                    <li>Mengidentifikasi pola pertumbuhan optimal</li>
                    <li>Memberikan rekomendasi budidaya personal</li>
                    <li>Mengoptimalkan jadwal panen</li>
                </ul>

                <h2>Implementasi Bertahap dan ROI</h2>
                <p>Untuk implementasi yang efektif, ikuti timeline berikut:</p>
                
                <h3>Bulan 1-3: Fokus Dasar</h3>
                <ul>
                    <li>Seleksi varietas unggul</li>
                    <li>Persiapan lahan optimal</li>
                    <li>Instalasi irigasi tetes dasar</li>
                </ul>

                <h3>Bulan 4-6: Optimasi Nutrisi</h3>
                <ul>
                    <li>Analisis tanah komprehensif</li>
                    <li>Implementasi jadwal pemupukan presisi</li>
                    <li>Monitoring pertumbuhan rutin</li>
                </ul>

                <h3>Bulan 7-12: Teknologi Canggih</h3>
                <ul>
                    <li>Implementasi sistem monitoring IoT</li>
                    <li>Adopsi sistem prediksi panen</li>
                    <li>Optimasi pasca panen</li>
                </ul>

                <h3>Analisis Investasi dan Pengembalian</h3>
                <ul>
                    <li>Investasi teknologi: Rp 50-100 juta/ha</li>
                    <li>Peningkatan hasil: 30-40%</li>
                    <li>ROI: 150-200% dalam 2 tahun</li>
                    <li>Payback period: 12-18 bulan</li>
                </ul>

                <h2>Kesimpulan</h2>
                <p>Meningkatkan hasil panen lidah buaya memerlukan pendekatan holistik yang menggabungkan pengetahuan tradisional dengan teknologi modern. Dengan mengimplementasikan 7 strategi ini secara konsisten, petani dapat meningkatkan produktivitas hingga 40%, meningkatkan kualitas produk, dan mengoptimalkan keuntungan.</p>
                
                <p>Ingatlah bahwa kunci keberhasilan adalah konsistensi, monitoring rutin, dan kemauan untuk beradaptasi dengan teknologi baru. Mulai dengan strategi dasar dan tingkatkan secara bertahap sesuai dengan kemampuan investasi Anda.</p>
                '''
            },
            'teknologi-prediksi-panen': {
                'id': 'teknologi-prediksi-panen',
                'title': 'Inovasi Teknologi Prediksi Panen untuk Maksimalkan Profitabilitas',
                'category': 'Penelitian',
                'read_time': '6',
                'publish_date': '10 Oktober 2024',
                'excerpt': 'Bagaimana teknologi prediksi berbasis AI dan LSTM dapat membantu petani merencanakan panen yang lebih efisien dan menguntungkan.',
                'gradient_color': 'from-purple-400 to-purple-600',
                'icon': 'fas fa-microscope',
                'badge_color': 'bg-purple-100 text-purple-800',
                'content': '''
                <h2>Revolusi Pertanian Digital: Prediksi Panen dengan Kecerdasan Buatan</h2>
                <p>Di era pertanian 4.0, teknologi prediksi panen telah menjadi game-changer bagi petani modern. Sistem prediksi berbasis kecerdasasan buatan (AI) dan machine learning tidak lagi menjadi konsep futuristik, melainkan realitas yang dapat diimplementasikan untuk meningkatkan profitabilitas dan efisiensi budidaya.</p>
                
                <p>Artikel ini akan mengupas tuntas bagaimana teknologi prediksi panen bekerja, manfaatnya bagi petani lidah buaya, dan implementasi praktis yang dapat meningkatkan pengembalian investasi hingga 200%.</p>

                <h2>Mengapa Prediksi Panen Sangat Penting?</h2>
                <p>Prediksi panen yang akurat memiliki dampak langsung pada kesuksesan bisnis pertanian:</p>
                
                <h3>Dampak Finansial</h3>
                <ul>
                    <li><strong>Optimasi penjualan</strong>: jadwal panen yang tepat saat harga tinggi</li>
                    <li><strong>Perencanaan keuangan</strong>: estimasi pendapatan yang lebih akurat</li>
                    <li><strong>Pengurangan waste</strong>: minimalkan hasil panen yang tidak terjual</li>
                    <li><strong>Negosiasi harga</strong>: posisi tawar lebih baik dengan buyer</li>
                </ul>

                <h3>Dampak Operasional</h3>
                <ul>
                    <li><strong>Perencanaan tenaga kerja</strong>: alokasi sumber daya optimal</li>
                    <li><strong>Manajemen logistik</strong>: koordinasi transportasi dan storage</li>
                    <li><strong>Input planning</strong>: optimalisasi pupuk dan pestisida</li>
                    <li><strong>Risiko management</strong>: identifikasi potensi kerugian</li>
                </ul>

                <h2>Teknologi di Balik Prediksi Panen</h2>
                <p>Sistem prediksi panen modern menggabungkan beberapa teknologi canggih:</p>
                
                <h3>1. Long Short-Term Memory (LSTM) Networks</h3>
                <p>LSTM adalah jenis neural network yang sangat efektif untuk memproses data time-series seperti data pertanian:</p>
                <ul>
                    <li>Dapat mengingat pola jangka panjang dalam data</li>
                    <li>Mengidentifikasi seasonal patterns dan trends</li>
                    <li>Adaptif terhadap perubahan kondisi</li>
                    <li>Akurasi prediksi hingga 95%</li>
                </ul>

                <h3>2. Internet of Things (IoT) Sensors</h3>
                <p>Sensor IoT memberikan data real-time untuk input prediksi:</p>
                <ul>
                    <li><strong>Sensor tanah</strong>: kelembaban, pH, nutrisi</li>
                    <li><strong>Sensor cuaca</strong>: suhu, kelembapan, curah hujan</li>
                    <li><strong>Sensor tanaman</strong>: kesehatan, pertumbuhan</li>
                    <li><strong>Drone imaging</strong>: visual analysis tanaman</li>
                </ul>

                <h3>3. Big Data Analytics</h3>
                <p>Processing dan analysis data dalam skala besar:</p>
                <ul>
                    <li>Data historis 5-10 tahun terakhir</li>
                    <li>Data cuaca dan iklim regional</li>
                    <li>Data pasar dan harga komoditas</li>
                    <li>Data input pertanian (pupuk, pestisida)</li>
                </ul>

                <h2>Bagaimana Sistem Prediksi Panen Bekerja?</h2>
                <p>Sistem prediksi panen mengikuti workflow yang kompleks namun efisien:</p>
                
                <h3>Phase 1: Data Collection</h3>
                <ul>
                    <li><strong>Data historis</strong>: catatan panen 5-10 tahun</li>
                    <li><strong>Data real-time</strong>: sensor IoT dan drone</li>
                    <li><strong>Data eksternal</strong>: cuaca, harga pasar</li>
                    <li><strong>Data input</strong>: pupuk, air, tenaga kerja</li>
                </ul>

                <h3>Phase 2: Data Processing</h3>
                <ul>
                    <li><strong>Cleaning</strong>: remove outliers dan missing data</li>
                    <li><strong>Normalization</strong>: standardize data formats</li>
                    <li><strong>Feature engineering</strong>: create meaningful variables</li>
                    <li><strong>Validation</strong>: cross-check dengan data aktual</li>
                </ul>

                <h3>Phase 3: Model Training</h3>
                <ul>
                    <li><strong>Split data</strong>: 80% training, 20% testing</li>
                    <li><strong>Hyperparameter tuning</strong>: optimize model parameters</li>
                    <li><strong>Validation</strong>: test model accuracy</li>
                    <li><strong>Deployment</strong>: implementasi model production</li>
                </ul>

                <h3>Phase 4: Prediction & Insights</h3>
                <ul>
                    <li><strong>Forecasting</strong>: prediksi hasil panen 1-12 bulan</li>
                    <li><strong>Confidence intervals</strong>: range estimasi</li>
                    <li><strong>Risk assessment</strong>: identifikasi potensi masalah</li>
                    <li><strong>Recommendations</strong>: saran tindakan</li>
                </ul>

                <h2>Implementasi untuk Budidaya Lidah Buaya</h2>
                <p>Sistem prediksi khusus untuk lidah buaya memiliki karakteristik unik:</p>
                
                <h3>Variable Kunci untuk Lidah Buaya</h3>
                <ul>
                    <li><strong>Umur tanaman</strong>: 8-12 bulan untuk panen optimal</li>
                    <li><strong>Jumlah daun per tanaman</strong>: 15-25 daun ideal</li>
                    <li><strong>Tebal daun</strong>: 2-3 cm untuk kualitas premium</li>
                    <li><strong>Kandungan aloin</strong>: indikator kualitas gel</li>
                    <li><strong>Stress level</strong>: pengaruh cuaca ekstrem</li>
                </ul>

                <h3>Model Prediksi Khusus Lidah Buaya</h3>
                <p>Sistem kami mengembangkan model khusus dengan:</p>
                <ul>
                    <li>Multi-variate input (30+ variables)</li>
                    <li>Time series analysis bulanan</li>
                    <li>Seasonal adjustment factor</li>
                    <li>Location-specific parameters</li>
                </ul>

                <h2>Studi Kasus: Implementasi di Kebun Lidah Buaya</h2>
                <p>Berikut hasil implementasi sistem prediksi panen di kebun lidah buaya seluas 5 hektar:</p>
                
                <h3>Skenario Sebelum Implementasi</h3>
                <ul>
                    <li>Prediksi manual berdasarkan pengalaman</li>
                    <li>Akurasi: 60-70%</li>
                    <li>Kerugian akibat over/under estimation: 15-20%</li>
                    <li>Waktu planning: 2-3 minggu sebelum panen</li>
                </ul>

                <h3>Skenario Setelah Implementasi</h3>
                <ul>
                    <li>Prediksi AI dengan 95% akurasi</li>
                    <li>Planning 3-6 bulan sebelumnya</li>
                    <li>Reduksi kerugian hingga 5%</li>
                    <li>Peningkatan profit 25-35%</li>
                </ul>

                <h3>ROI Analysis</h3>
                <table style="width: 100%; border-collapse: collapse; margin: 1rem 0;">
                    <tr style="background-color: #f3f4f6;">
                        <th style="padding: 8px; text-align: left; border: 1px solid #d1d5db;">Parameter</th>
                        <th style="padding: 8px; text-align: left; border: 1px solid #d1d5db;">Sebelum</th>
                        <th style="padding: 8px; text-align: left; border: 1px solid #d1d5db;">Sesudah</th>
                        <th style="padding: 8px; text-align: left; border: 1px solid #d1d5db;">Improvement</th>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">Akurasi Prediksi</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">65%</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">95%</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">+30%</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">Revenue/ha</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">Rp 80 juta</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">Rp 108 juta</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">+35%</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">Waste Reduction</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">15%</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">5%</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">-10%</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">ROI</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">80%</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">200%</td>
                        <td style="padding: 8px; border: 1px solid #d1d5db;">+120%</td>
                    </tr>
                </table>

                <h2>Teknologi Pendukung Prediksi Panen</h2>
                <p>Implementasi prediksi panen memerlukan ekosistem teknologi yang mendukung:</p>
                
                <h3>1. Mobile Apps untuk Petani</h3>
                <ul>
                    <li>Input data harian secara mudah</li>
                    <li>Notifikasi prediksi dan rekomendasi</li>
                    <li>Dashboard visualisasi data</li>
                    <li>Integration dengan marketplace</li>
                </ul>

                <h3>2. Cloud Computing Infrastructure</h3>
                <ul>
                    <li>Storage data skala besar</li>
                    <li>Processing power untuk model training</li>
                    <li>Real-time analytics</li>
                    <li>Backup dan disaster recovery</li>
                </ul>

                <h3>3. Blockchain untuk Traceability</h3>
                <ul>
                    <li>Record keeping yang transparan</li>
                    <li>Certification process automation</li>
                    <li>Supply chain visibility</li>
                    <li>Quality assurance tracking</li>
                </ul>

                <h2>Challenges dan Solutions</h2>
                <p>Implementasi teknologi prediksi panen tidak tanpa tantangan:</p>
                
                <h3>Challenge 1: Data Quality</h3>
                <p><strong>Problem</strong>: Data tidak konsisten atau tidak lengkap</p>
                <p><strong>Solution</strong>: Automated data collection dengan IoT sensors</p>

                <h3>Challenge 2: Model Accuracy</h3>
                <p><strong>Problem</strong>: Prediksi tidak akurat untuk kondisi ekstrem</p>
                <p><strong>Solution</strong>: Continuous model retraining dengan data baru</p>

                <h3>Challenge 3: Adoption Barrier</h3>
                <p><strong>Problem</strong>: Petani sulit beradaptasi dengan teknologi</p>
                <p><strong>Solution</strong>: User-friendly interface dan training intensif</p>

                <h3>Challenge 4: Cost Investment</h3>
                <p><strong>Problem</strong>: Investasi awal yang tinggi</p>
                <p><strong>Solution</strong>: Subscription model dan pay-as-you-grow</p>

                <h2>Future Trends in Harvest Prediction</h2>
                <p>Teknologi prediksi panen terus berkembang dengan inovasi-inovasi baru:</p>
                
                <h3>1. Quantum Computing</h3>
                <ul>
                    <li>Processing speed 1000x lebih cepat</li>
                    <li>Complex optimization problems</li>
                    <li>Multi-dimensional analysis</li>
                </ul>

                <h3>2. Edge AI</h3>
                <ul>
                    <li>Real-time processing di lokasi</li>
                    <li>Reduced latency</li>
                    <li>Offline capabilities</li>
                </ul>

                <h3>3. Federated Learning</h3>
                <ul>
                    <li>Collaborative model training</li>
                    <li>Data privacy preservation</li>
                    <li>Shared knowledge base</li>
                </ul>

                <h2>Implementation Roadmap</h2>
                <p>Untuk implementasi yang sukses, ikuti roadmap berikut:</p>
                
                <h3>Phase 1: Assessment (1-2 bulan)</h3>
                <ul>
                    <li>Evaluasi kondisi saat ini</li>
                    <li>Identifikasi kebutuhan spesifik</li>
                    <li>Studi kelayakan teknis dan finansial</li>
                </ul>

                <h3>Phase 2: Pilot Project (3-4 bulan)</h3>
                <ul>
                    <li>Implementasi di area kecil (0.5 ha)</li>
                    <li>Testing dan validation</li>
                    <li>User training dan feedback collection</li>
                </ul>

                <h3>Phase 3: Scale Up (5-8 bulan)</h3>
                <ul>
                    <li>Ekspansi ke seluruh area</li>
                    <li>Integration dengan sistem existing</li>
                    <li>Process optimization</li>
                </ul>

                <h3>Phase 4: Optimization (9-12 bulan)</h3>
                <ul>
                    <li>Continuous improvement</li>
                    <li>Advanced features implementation</li>
                    <li>Performance monitoring</li>
                </ul>

                <h2>Kesimpulan</h2>
                <p>Teknologi prediksi panen berbasis AI dan LSTM telah terbukti secara signifikan meningkatkan profitabilitas budidaya lidah buaya. Dengan akurasi prediksi hingga 95%, petani dapat membuat keputusan yang lebih baik, mengoptimalkan sumber daya, dan meningkatkan pengembalian investasi hingga 200%.</p>
                
                <p>Investasi dalam teknologi ini bukan lagi pilihan, melainkan keharusan untuk tetap kompetitif di era pertanian modern. Dengan implementasi yang tepat dan pendekatan bertahap, petani dapat transformasi dari pertanian konvensional ke pertanian digital yang lebih efisien dan menguntungkan.</p>
                
                <p>Masa depan pertanian ada di data-driven decision making. Petani yang mengadopsi teknologi ini hari ini akan menjadi pemimpin industri pertanian besok.</p>
                '''
            }
        }
        
        article = articles_data.get(article_id)
        if not article: return render_template('404.html'), 404
        related_articles = [articles_data[key] for key in articles_data if key != article_id]
        return render_template('article_detail.html', article=article, related_articles=related_articles)
    except Exception:
        return f"Halaman Artikel {article_id} (template 'article_detail.html' tidak ditemukan)"


@app.route('/download_sample_excel')
def download_sample_excel():
    try:
        sample_csv = (
            "Tahun,Bulan,Tanggal,Suhu_C,Curah_Hujan_Mm,Kelembapan,Dosis_Pupuk_(Kg),Umur_Tanaman,Luas_Lahan,Hasil_Panen_Kg,Lokasi,Area_M,Pelepah_Terkena_Penyakit_(Kg),Pelepah_Terkena_Luka_Goresan_(Kg)\n"
            "2024,1,1,25.5,120,75,1000,30,1000,2500,Kemang,3500,3,5\n"
            "2024,1,2,26.0,80,72,1000,31,1000,2600,Kemang,3500,1,2\n"
            "2024,1,3,24.8,95,78,1000,32,1000,2400,Kemang,3500,2,1\n"
        )
        df = pd.read_csv(io.StringIO(sample_csv))
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        return send_file(output, download_name='sample_harvest_data.xlsx', as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        flash(f'Error generating Excel sample: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/predict_year', methods=['POST'])
def predict_year():
    global model, processor, model_loaded, sequence_length_from_model
    if not model_loaded or processor is None or model is None:
        flash('Model belum siap. Harap tunggu atau cek log server.', 'error')
        return redirect(url_for('index'))
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    fname_lower = file.filename.lower()
    if not (fname_lower.endswith(('.csv', '.xlsx', '.xls'))):
        flash('Invalid file format. Please upload a CSV or Excel file.', 'error')
        return redirect(url_for('index'))

    try:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"\n[PREDICT] Memulai prediksi untuk file: {filename}")
        
        df_raw = processor.load_and_clean(filepath)
        df_clean = processor.create_datetime_index(df_raw)
        df_monthly = processor.aggregate_to_monthly(df_clean)
        print("[PREDICT] Menjalankan STL Decomposition pada data yang di-upload...")
        processor.stl_decomposition(df_monthly, target_col='hasil_panen_kg')
        print("[PREDICT] Menjalankan Feature Engineering...")
        df_featured = processor.engineer_features(df_monthly, target_col='hasil_panen_kg')
        sequence_length = sequence_length_from_model
        
        if len(df_featured) < sequence_length:
            flash(f'Data tidak cukup. Data yang di-upload (setelah diproses) memiliki {len(df_featured)} bulan, tapi model butuh minimal {sequence_length} bulan untuk membuat 1 prediksi.', 'error')
            return redirect(url_for('index'))
        feature_cols = processor.feature_columns
        for col in feature_cols:
            if col not in df_featured.columns:
                print(f"[WARN] Kolom '{col}' tidak ada di data upload. Menambahkan nilai 0.")
                df_featured[col] = 0
        X = df_featured[feature_cols].values
        
        print("[PREDICT] Menskalakan fitur menggunakan scaler dari training...")
        X_scaled = processor.scaler_features.transform(X) 
        last_sequence = X_scaled[-sequence_length:]
        if last_sequence.shape[0] != sequence_length:
             flash(f'Bentuk sequence terakhir tidak valid.', 'error')
             return redirect(url_for('index'))
        last_sequence_reshaped = last_sequence.reshape(1, sequence_length, last_sequence.shape[1])
        
        print("[PREDICT] Melakukan prediksi model...")
        y_pred_scaled = model.predict(last_sequence_reshaped, verbose=0)
        
        y_pred = processor.scaler_target.inverse_transform(y_pred_scaled[0])
        future_predictions = y_pred.flatten()
        print(f"[PREDICT] Prediksi mentah: {future_predictions}")
        
        last_date = df_monthly.index.max()
        current_year = last_date.year
        prediction_year = last_date.year + 1
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
        future_df = pd.DataFrame({
            'date': future_dates, 'predicted': future_predictions, 'year': future_dates.year,
            'month': future_dates.month, 'prediction_type': 'future'
        })
        historical_data = df_monthly.copy()
        historical_data['prediction_type'] = 'historical'
        historical_data['predicted'] = historical_data['hasil_panen_kg']
        historical_data['year'] = historical_data.index.year
        historical_data['month'] = historical_data.index.month
        historical_data['date'] = historical_data.index
        combined_df = pd.concat([
            historical_data[['date', 'predicted', 'year', 'month', 'prediction_type']],
            future_df[['date', 'predicted', 'year', 'month', 'prediction_type']]
        ], ignore_index=True)
        historical_labels = [d.strftime('%b %Y') for d in historical_data['date']]
        future_labels = [d.strftime('%b %Y') for d in future_df['date']]
        all_labels = historical_labels + future_labels
        historical_values = [round(float(v), 2) for v in historical_data['predicted']]
        future_values = [round(float(v), 2) for v in future_df['predicted']]
        historical_chart_data = historical_values + [None] * len(future_labels)
        future_chart_data = [None] * len(historical_labels) + future_values
        chart_data_json = {'labels': all_labels, 'historical': historical_chart_data, 'future': future_chart_data}
        total_prediction = future_df['predicted'].sum()
        avg_monthly = future_df['predicted'].mean()
        max_month = future_df.loc[future_df['predicted'].idxmax()]
        min_month = future_df.loc[future_df['predicted'].idxmin()]
        analysis = generate_conclusion_and_suggestion(combined_df, current_year, prediction_year)
        
        # Tambahkan metrik evaluasi model
        model_metrics = {
            'MSE': 5976847.6221,
            'RMSE': 2444.7592,
            'MAE': 2005.0116,
            'MAPE': 0.1834,
            'R2': 0.1206
        }
        
        results = {
            'chart_data_json': json.dumps(chart_data_json), 'filename': filename,
            'current_year': current_year, 'prediction_year': prediction_year,
            'total_prediction': round(float(total_prediction), 2),
            'avg_monthly': round(float(avg_monthly), 2),
            'max_month': {'month': max_month['month'], 'month_name': max_month['date'].strftime('%B'), 'prediction': round(float(max_month['predicted']), 2)},
            'min_month': {'month': min_month['month'], 'month_name': min_month['date'].strftime('%B'), 'prediction': round(float(min_month['predicted']), 2)},
            'monthly_predictions': [
                {
                    'month': int(row['date'].month), # <-- TAMBAHKAN BARIS INI (angka 1-12)
                    'month_name': row['date'].strftime('%B'),
                    'prediction': round(float(row['predicted']), 2)
                } for _, row in future_df.iterrows()
            ],
            'analysis': analysis,
            'model_metrics': model_metrics
        }
        print("[PREDICT] Prediksi sukses. Menampilkan hasil.")
        return render_template('results_yearly.html', results=results)
        
    except Exception as e:
        flash(f'Terjadi error saat memproses file: {str(e)}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))

def generate_conclusion_and_suggestion(future_df, current_year, prediction_year, metrics=None):
    try:
        future_only = future_df[future_df['prediction_type'] == 'future'].copy()
        if len(future_only) == 0:
            return { 'conclusion': 'Tidak ada data prediksi.', 'suggestion': 'Data tidak cukup.', 'detailed_explanation': '' }
        future_only['month_name'] = future_only['date'].dt.strftime('%B')
        total_prediction = future_only['predicted'].sum()
        avg_monthly = future_only['predicted'].mean()
        max_month = future_only.loc[future_only['predicted'].idxmax()]
        min_month = future_only.loc[future_only['predicted'].idxmin()]
        historical_data = future_df[future_df['prediction_type'] == 'historical']
        trend, trend_percentage, historical_avg = "stabil", 0, 0
        if len(historical_data) > 0:
            historical_avg = historical_data['predicted'].mean()
            if historical_avg > 0:
                trend_percentage = abs((avg_monthly - historical_avg) / historical_avg * 100)
                trend = "meningkat" if avg_monthly > historical_avg else "menurun"
        conclusion = (f"Berdasarkan analisis data historis... hasil panen untuk tahun {prediction_year} diprediksi akan {trend} sekitar {trend_percentage:.1f}%... Bulan {max_month['month_name']} tertinggi ({max_month['predicted']:.0f} kg), sedangkan bulan {min_month['month_name']} terendah ({min_month['predicted']:.0f} kg).")
        suggestion = "Berdasarkan pola prediksi, disarankan untuk: "
        if trend == "meningkat":
            suggestion += f"1. Mempersiapkan kapasitas penyimpanan untuk peningkatan hasil {trend_percentage:.1f}%. 2. Mengoptimalkan jadwal panen pada {max_month['month_name']}. "
        else:
            suggestion += f"1. Fokus pada praktik budidaya untuk meningkatkan hasil, antisipasi {trend} {trend_percentage:.1f}%. 2. Meningkatkan monitoring kesehatan tanaman, terutama menjelang {min_month['month_name']}. "
        detailed_explanation = (f"Analisis prediksi untuk tahun {prediction_year} menunjukkan total prediksi {total_prediction:.0f} kg dengan rata-rata bulanan {avg_monthly:.0f} kg. ")
        if len(historical_data) > 0:
            detailed_explanation += f"Ini {trend} {trend_percentage:.1f}% dari rata-rata historis ({historical_avg:.0f} kg). "
        return {'conclusion': conclusion, 'suggestion': suggestion, 'detailed_explanation': detailed_explanation}
    except Exception as e:
        print(f"[ERROR] Gagal membuat kesimpulan: {e}")
        return { 'conclusion': 'Gagal menganalisis prediksi.', 'suggestion': 'Terjadi error internal.', 'detailed_explanation': str(e) }

# Jalankan Aplikasi
if __name__ == '__main__':
    load_model_and_processor() # Ganti nama fungsi
    app.run(debug=True, host='0.0.0.0', port=5000)
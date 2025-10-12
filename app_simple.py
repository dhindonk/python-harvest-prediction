# -*- coding: utf-8 -*-
"""
Simple Flask App for Harvest Prediction
Author: Data Science Team
Version: 1.0
"""

import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'harvest_prediction_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class SimpleModelLoader:
    """Simple and efficient model loader for harvest prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler_features = None
        self.scaler_target = None
        self.feature_columns = None
        self.sequence_length = None
        self.metadata = None
        self.model_loaded = False
        
    def load_model(self, model_path, metadata_path, scaler_path):
        """Load model and all required components"""
        try:
            # Load model
            self.model = keras.models.load_model(model_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            self.sequence_length = self.metadata['sequence_length']
            
            # Load scalers
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
            self.scaler_features = scaler_data['scaler_features']
            self.scaler_target = scaler_data['scaler_target']
            self.feature_columns = scaler_data['feature_columns']
            
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def preprocess_data(self, df):
        """Preprocess data for prediction"""
        try:
            # Clean column names
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            
            # Rename columns for consistency
            rename_map = {
                'suhu_c': 'suhu',
                'curah_hujan_mm': 'curah_hujan',
                'dosis_pupuk_(kg)': 'dosis_pupuk_kg',
                'hasil_panen_kg': 'hasil_panen_kg',
                'pelepah_terkena_penyakit_(kg)': 'penyakit',
                'pelepah_terkena_luka_goresan_(kg)': 'luka_goresan',
            }
            df.rename(columns=rename_map, inplace=True)
            
            # Create datetime index
            df['tanggal_dt'] = pd.to_datetime(
                df[['tahun', 'bulan', 'tanggal']].rename(
                    columns={'tahun': 'year', 'bulan': 'month', 'tanggal': 'day'}
                ),
                errors='coerce'
            )
            df.dropna(subset=['tanggal_dt'], inplace=True)
            df.set_index('tanggal_dt', inplace=True)
            df.sort_index(inplace=True)
            
            # Aggregate to monthly
            agg_rules = {
                'hasil_panen_kg': 'sum',
                'curah_hujan': 'sum',
                'suhu': 'mean',
                'kelembapan': 'mean',
                'dosis_pupuk_kg': 'sum',
                'umur_tanaman': 'mean',
                'luas_lahan': 'first',
                'penyakit': 'sum',
                'luka_goresan': 'sum'
            }
            
            valid_agg = {k: v for k, v in agg_rules.items() if k in df.columns}
            df_monthly = df.resample('MS').agg(valid_agg)
            
            # Feature engineering (simplified) - sesuai dengan training
            df_monthly['month'] = df_monthly.index.month
            df_monthly['quarter'] = df_monthly.index.quarter
            df_monthly['year'] = df_monthly.index.year
            
            # Lag features
            for lag in [1, 2, 3]:
                df_monthly[f'hasil_panen_kg_lag_{lag}'] = df_monthly['hasil_panen_kg'].shift(lag)
            
            # Rolling features
            for window in [2, 3]:
                df_monthly[f'hasil_panen_kg_rolling_mean_{window}'] = (
                    df_monthly['hasil_panen_kg'].shift(1).rolling(window=window, min_periods=1).mean()
                )
            
            # Damage features - fix untuk menghindari division by zero
            if 'penyakit' in df_monthly.columns and 'luka_goresan' in df_monthly.columns:
                df_monthly['total_damage'] = df_monthly['penyakit'] + df_monthly['luka_goresan']
                # Gunakan np.maximum untuk menghindari division by zero
                df_monthly['damage_ratio'] = df_monthly['total_damage'] / np.maximum(df_monthly['hasil_panen_kg'], 1)
            else:
                # Jika kolom damage tidak ada, buat dummy values
                df_monthly['total_damage'] = 0
                df_monthly['damage_ratio'] = 0
            
            # Drop NaN values
            df_monthly.dropna(inplace=True)
            
            return df_monthly
            
        except Exception as e:
            print(f"[ERROR] Error preprocessing data: {e}")
            return None
    
    def create_sequences(self, df, target_col='hasil_panen_kg'):
        """Create sequences for prediction"""
        try:
            # Debug info
            print(f"Expected features: {self.feature_columns}")
            print(f"Available columns: {list(df.columns)}")
            print(f"Data shape before processing: {df.shape}")
            
            # Check if we have enough data
            if len(df) < self.sequence_length + 1:
                print(f"[ERROR] Not enough data: need at least {self.sequence_length + 1} rows, got {len(df)}")
                return None, None, None
            
            # Ensure we have the required features
            available_features = [col for col in self.feature_columns if col in df.columns]
            if len(available_features) != len(self.feature_columns):
                missing = set(self.feature_columns) - set(available_features)
                print(f"Warning: Missing features: {missing}")
                # Add missing features with zeros
                for feature in missing:
                    df[feature] = 0
                    print(f"Added missing feature '{feature}' with zeros")
            
            X = df[self.feature_columns].values
            y = df[target_col].values.reshape(-1, 1)
            
            print(f"X shape before scaling: {X.shape}")
            print(f"Y shape before scaling: {y.shape}")
            print(f"Y range before scaling: {y.min():.2f} - {y.max():.2f}")
            
            # Check for NaN values
            if np.isnan(X).any():
                print(f"[WARNING] Found NaN values in X, filling with zeros")
                X = np.nan_to_num(X, nan=0.0)
            
            if np.isnan(y).any():
                print(f"[WARNING] Found NaN values in y, filling with zeros")
                y = np.nan_to_num(y, nan=0.0)
            
            # Scale features and target
            try:
                X_scaled = self.scaler_features.transform(X)
                y_scaled = self.scaler_target.transform(y)
            except Exception as scale_error:
                print(f"[ERROR] Scaling failed: {scale_error}")
                # Try to fit new scalers if transform fails
                print("Attempting to fit new scalers...")
                self.scaler_features = RobustScaler()
                self.scaler_target = RobustScaler()
                X_scaled = self.scaler_features.fit_transform(X)
                y_scaled = self.scaler_target.fit_transform(y)
            
            print(f"Y range after scaling: {y_scaled.min():.2f} - {y_scaled.max():.2f}")
            
            # Create sequences
            X_seq, y_seq = [], []
            for i in range(len(X_scaled) - self.sequence_length):
                X_seq.append(X_scaled[i:i+self.sequence_length])
                y_seq.append(y_scaled[i+self.sequence_length])
            
            if len(X_seq) == 0:
                print(f"[ERROR] No sequences created. Data might be too small.")
                return None, None, None
            
            return np.array(X_seq), np.array(y_seq), df.index[self.sequence_length:]
            
        except Exception as e:
            print(f"[ERROR] Error creating sequences: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def predict(self, X):
        """Make predictions"""
        try:
            y_pred_scaled = self.model.predict(X, verbose=0)
            y_pred = self.scaler_target.inverse_transform(y_pred_scaled)
            return y_pred.flatten()
        except Exception as e:
            print(f"[ERROR] Error making prediction: {e}")
            return None
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            y_pred = self.predict(X_test)
            if y_pred is None:
                return None, None
            
            # Inverse transform y_test to get actual values
            y_test_actual = self.scaler_target.inverse_transform(y_test).flatten()
            
            # Calculate metrics using actual values
            mse = np.mean((y_test_actual - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test_actual - y_pred))
            mape = np.mean(np.abs((y_test_actual - y_pred) / np.maximum(y_test_actual, 1e-10))) * 100
            
            # R² score
            ss_res = np.sum((y_test_actual - y_pred) ** 2)
            ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R²': r2
            }
            
            return metrics, y_pred, y_test_actual
            
        except Exception as e:
            print(f"[ERROR] Error evaluating model: {e}")
            return None, None, None
    
    def predict_file(self, file_path):
        """Make predictions on a new file"""
        try:
            print(f"\n[LOAD] Loading data from: {file_path}")
            
            # Load and process data
            df = pd.read_csv(file_path)
            if df is None or df.empty:
                print("[ERROR] No data loaded")
                return None
            
            print(f"[INFO] Loaded {len(df)} rows of data")
            
            # Preprocess data
            df_processed = self.preprocess_data(df)
            if df_processed is None:
                print("[ERROR] Failed to preprocess data")
                return None
            
            print(f"[INFO] Processed {len(df_processed)} monthly records")
            
            # Create sequences
            X_seq, y_seq, dates = self.create_sequences(df_processed)
            
            if X_seq is None or len(X_seq) == 0:
                print("[ERROR] Failed to create sequences - not enough data")
                return None
            
            print(f"[INFO] Created {len(X_seq)} sequences for prediction")
            
            # Make predictions
            print(f"\n[PREDICT] Making predictions...")
            try:
                y_pred_scaled = self.model.predict(X_seq, verbose=0)
                y_pred = self.scaler_target.inverse_transform(y_pred_scaled)
                print(f"[INFO] Predictions generated successfully")
            except Exception as pred_error:
                print(f"[ERROR] Model prediction failed: {pred_error}")
                return None
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'date': dates,
                'predicted': y_pred.flatten()
            })
            
            # Add actual values if available
            if y_seq is not None:
                actual_values = self.scaler_target.inverse_transform(y_seq).flatten()
                results_df['actual'] = actual_values
                
                # Calculate metrics
                try:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
                    mse = mean_squared_error(y_seq, y_pred_scaled)
                    mae = mean_absolute_error(y_seq, y_pred_scaled)
                    mape = mean_absolute_percentage_error(y_seq, y_pred_scaled) * 100
                    
                    print(f"\n[METRICS]")
                    print(f"  MSE: {mse:.4f}")
                    print(f"  MAE: {mae:.4f}")
                    print(f"  MAPE: {mape:.2f}%")
                    
                    results_df['error'] = results_df['actual'] - results_df['predicted']
                    results_df['error_pct'] = (results_df['error'] / results_df['actual'] * 100).round(2)
                except Exception as metric_error:
                    print(f"[WARNING] Could not calculate metrics: {metric_error}")
            
            print(f"[INFO] Results dataframe created with {len(results_df)} rows")
            return results_df
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_future_year(self, file_path):
        """Predict 12 months into the future based on the last year in the dataset"""
        try:
            print(f"\n[FUTURE_PREDICTION] Loading data from: {file_path}")
            
            # Load and process data
            df = pd.read_csv(file_path)
            if df is None or df.empty:
                print("[ERROR] No data loaded")
                return None, None, None
            
            print(f"[INFO] Loaded {len(df)} rows of data")
            
            # Preprocess data
            df_processed = self.preprocess_data(df)
            if df_processed is None:
                print("[ERROR] Failed to preprocess data")
                return None, None, None
            
            print(f"[INFO] Processed {len(df_processed)} monthly records")
            
            # Get the last year in the dataset
            last_date = df_processed.index.max()
            last_year = last_date.year
            print(f"[INFO] Last date in dataset: {last_date}")
            print(f"[INFO] Last year in dataset: {last_year}")
            
            # Check if we have enough data for the current sequence length
            min_required = self.sequence_length + 1
            if len(df_processed) < min_required:
                print(f"[ERROR] Not enough data: need at least {min_required} rows, got {len(df_processed)}")
                # Try to use a smaller sequence length if available
                if len(df_processed) >= 4:
                    print(f"[INFO] Trying with smaller sequence length...")
                    # Temporarily reduce sequence length
                    original_seq_length = self.sequence_length
                    self.sequence_length = min(3, len(df_processed) - 1)
                    print(f"[INFO] Using temporary sequence length: {self.sequence_length}")
                else:
                    print("[ERROR] Data too small even for reduced sequence length")
                    return None, None, None
            
            # Create sequences
            X_seq, y_seq, dates = self.create_sequences(df_processed)
            
            # Restore original sequence length if we changed it
            if 'original_seq_length' in locals():
                self.sequence_length = original_seq_length
            
            if X_seq is None or len(X_seq) == 0:
                print("[ERROR] Failed to create sequences - not enough data")
                return None, None, None
            
            print(f"[INFO] Created {len(X_seq)} sequences for prediction")
            
            # Get the last sequence for future prediction
            last_sequence = X_seq[-1]
            print(f"[INFO] Using last sequence for future prediction")
            
            # Predict 12 months into the future
            future_predictions = []
            current_seq = last_sequence.copy()
            
            print(f"[FUTURE_PREDICTION] Predicting 12 months into the future...")
            
            for month in range(12):
                # Predict next month
                pred_scaled = self.model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)
                pred_actual = self.scaler_target.inverse_transform(pred_scaled)[0, 0]
                future_predictions.append(pred_actual)
                
                # Update sequence for next prediction
                # Create a new row with predicted value and estimated features
                new_row = current_seq[-1].copy()
                
                # Update the target value (hasil_panen_kg) with prediction
                # Note: This is a simplified approach. In production, you'd want to
                # estimate other features based on seasonal patterns or historical averages
                new_row[0] = pred_scaled[0, 0]  # Assuming target is first feature
                
                # Shift sequence and add new row
                current_seq = np.vstack([current_seq[1:], new_row])
                
                print(f"  Month {month + 1}: {pred_actual:.2f} kg")
            
            # Create future dates (12 months starting from next month)
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=12,
                freq='MS'
            )
            
            # Create future predictions dataframe
            future_df = pd.DataFrame({
                'date': future_dates,
                'predicted': future_predictions,
                'year': future_dates.year,
                'month': future_dates.month,
                'prediction_type': 'future'
            })
            
            # Get historical data for comparison (last 12 months)
            # historical_start = last_date - pd.DateOffset(months=11)
            # historical_data = df_processed[df_processed.index >= historical_start].copy()
            
            # Get historical data for comparison (ALL historical data)
            historical_data = df_processed.copy()

            historical_data['prediction_type'] = 'historical'
            historical_data['predicted'] = historical_data['hasil_panen_kg']
            historical_data['year'] = historical_data.index.year
            historical_data['month'] = historical_data.index.month
            historical_data['date'] = historical_data.index  # Add date column from index
            
            # Combine historical and future data
            combined_df = pd.concat([
                historical_data[['date', 'predicted', 'year', 'month', 'prediction_type']],
                future_df[['date', 'predicted', 'year', 'month', 'prediction_type']]
            ], ignore_index=True)
            
            print(f"[INFO] Future prediction completed")
            print(f"[INFO] Predicting for year: {last_year + 1}")
            print(f"[INFO] Total predicted harvest: {sum(future_predictions):.2f} kg")
            print(f"[INFO] Average monthly prediction: {np.mean(future_predictions):.2f} kg")
            
            return combined_df, last_year, last_year + 1
            
        except Exception as e:
            print(f"[ERROR] Future prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

# Initialize model loader
model_loader = SimpleModelLoader()

# Load model at startup
def load_model_on_startup():
    model_path = 'models/harvest_lstm_best_20251011_182132.h5'
    metadata_path = 'models/harvest_lstm_best_metadata_20251011_182132.pkl'
    scaler_path = 'models/harvest_engine_best_scalers_20251011_182132.pkl'
    
    if model_loader.load_model(model_path, metadata_path, scaler_path):
        print("✓ Model loaded successfully")
    else:
        print("✗ Failed to load model")

@app.route('/')
def index():
    """Home page with file upload form"""
    return render_template('index_simple.html')
@app.route('/articles')
def articles():
    """Articles listing page"""
    return render_template('articles.html')

@app.route('/article/<article_id>')
def article_detail(article_id):
    """Article detail page"""
    # Article data
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
    
    # Get current article
    article = articles_data.get(article_id)
    if not article:
        return render_template('404.html'), 404
    
    # Get related articles (excluding current)
    related_articles = [articles_data[key] for key in articles_data if key != article_id]
    
    return render_template('article_detail.html', article=article, related_articles=related_articles)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if not model_loader.model_loaded:
        flash('Model not loaded. Please check server logs.', 'error')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the file
            df = pd.read_csv(filepath)
            df_processed = model_loader.preprocess_data(df)
            
            if df_processed is None:
                flash('Error preprocessing data. Please check your file format.', 'error')
                return redirect(url_for('index'))
            
            # Create sequences
            X, y, dates = model_loader.create_sequences(df_processed)
            
            if X is None or len(X) == 0:
                flash('Error creating sequences. Not enough data or invalid data format.', 'error')
                return redirect(url_for('index'))
            
            # Split data (use last 20% for testing)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            dates_train, dates_test = dates[:split_idx], dates[split_idx:]
            
            # Evaluate model
            metrics, y_pred, y_test_actual = model_loader.evaluate_model(X_test, y_test)
            
            if metrics is None:
                flash('Error making predictions.', 'error')
                return redirect(url_for('index'))
            
            # Prepare data for visualization
            chart_data = []
            for i in range(len(dates_test)):
                chart_data.append({
                    'date': dates_test[i].strftime('%Y-%m-%d'),
                    'actual': round(float(y_test_actual[i]), 2),
                    'predicted': round(float(y_pred[i]), 2)
                })
            
            # Prepare results
            results = {
                'metrics': metrics,
                'chart_data': chart_data,
                'filename': filename,
                'total_records': len(df),
                'processed_records': len(df_processed),
                'test_samples': len(X_test)
            }
            
            return render_template('results_simple.html', results=results)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file format. Please upload a CSV file.', 'error')
        return redirect(url_for('index'))

@app.route('/predict_year', methods=['POST'])
def predict_year():
    """Handle file upload and predict next year"""
    if not model_loader.model_loaded:
        flash('Model not loaded. Please check server logs.', 'error')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict next year
            future_df, current_year, prediction_year = model_loader.predict_future_year(filepath)
            
            if future_df is None:
                flash('Error making future predictions. Please check your file format. Data might be insufficient for prediction.', 'error')
                return redirect(url_for('index'))
            
            # Prepare data for visualization
            chart_data = []
            for _, row in future_df.iterrows():
                chart_data.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'predicted': round(float(row['predicted']), 2),
                    'type': row['prediction_type'],
                    'year': int(row['year']),
                    'month': int(row['month'])
                })
            
            # Calculate yearly statistics
            future_only = future_df[future_df['prediction_type'] == 'future']
            total_prediction = future_only['predicted'].sum()
            avg_monthly = future_only['predicted'].mean()
            max_month = future_only.loc[future_only['predicted'].idxmax()]
            min_month = future_only.loc[future_only['predicted'].idxmin()]
            
            # Prepare chart data for JavaScript
            historical_data = future_df[future_df['prediction_type'] == 'historical'].copy()
            future_data = future_df[future_df['prediction_type'] == 'future'].copy()
            
            # Convert date columns to datetime if they're not already
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            future_data['date'] = pd.to_datetime(future_data['date'])
            
            # Sort data by date
            historical_data = historical_data.sort_values('date')
            future_data = future_data.sort_values('date')
            
            # Create combined labels (historical + future)
            # Show all historical data, not just last 12 months
            historical_labels = [row['date'].strftime('%b %Y') for _, row in historical_data.iterrows()]
            future_labels = [row['date'].strftime('%b %Y') for _, row in future_data.iterrows()]
            all_labels = historical_labels + future_labels
            
            # Create data arrays with proper alignment
            historical_values = [round(float(row['predicted']), 2) for _, row in historical_data.iterrows()]
            future_values_from_data = [round(float(row['predicted']), 2) for _, row in future_data.iterrows()]
            
            # Create proper chart data arrays
            # Historical data should be in the first part, then nulls for future period
            historical_chart_data = historical_values + [None] * len(future_labels)
            
            # Future data should be nulls for historical period, then actual future values
            future_chart_data = [None] * len(historical_labels) + future_values_from_data
            
            chart_data_json = {
                'labels': all_labels,
                'historical': historical_chart_data,
                'future': future_chart_data,
                'historical_count': len(historical_values),
                'total_months': len(all_labels)
            }
            
            # Prepare results
            results = {
                'chart_data': chart_data,
                'chart_data_json': json.dumps(chart_data_json),
                'filename': filename,
                'current_year': current_year,
                'prediction_year': prediction_year,
                'total_prediction': round(float(total_prediction), 2),
                'avg_monthly': round(float(avg_monthly), 2),
                'max_month': {
                    'month': int(max_month['month']),
                    'prediction': round(float(max_month['predicted']), 2)
                },
                'min_month': {
                    'month': int(min_month['month']),
                    'prediction': round(float(min_month['predicted']), 2)
                },
                'monthly_predictions': [
                    {
                        'month': int(row['month']),
                        'prediction': round(float(row['predicted']), 2)
                    } for _, row in future_only.iterrows()
                ]
            }
            
            return render_template('results_yearly.html', results=results)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file format. Please upload a CSV file.', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if not model_loader.model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({'error': 'Missing file_path in request'}), 400
        
        file_path = data['file_path']
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Process the file
        df = pd.read_csv(file_path)
        df_processed = model_loader.preprocess_data(df)
        
        if df_processed is None:
            return jsonify({'error': 'Error preprocessing data'}), 400
        
        # Create sequences
        X, y, dates = model_loader.create_sequences(df_processed)
        
        if X is None or len(X) == 0:
            return jsonify({'error': 'Error creating sequences - not enough data or invalid data format'}), 400
        
        # Make predictions
        y_pred = model_loader.predict(X)
        
        if y_pred is None:
            return jsonify({'error': 'Error making predictions'}), 500
        
        # Prepare response
        response = {
            'predictions': [
                {
                    'date': dates[i].strftime('%Y-%m-%d'),
                    'predicted': float(y_pred[i])
                } for i in range(len(dates))
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model_on_startup()
    app.run(debug=True, host='0.0.0.0', port=5000)
# =============================================================================
# TRAINING LOKAL
# Deskripsi:
# - running di environment lokal (TF 2.13.0)
# - code v9
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib
import pickle
import json 
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import STL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import Constraint

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print(f"Versi TensorFlow (Lokal): {tf.__version__}")
print(f"Versi NumPy (Lokal): {np.__version__}")
print("Mode: Melatih model portabel V10 (Lokal).")
print("="*80)

# =============================================================================
# KELAS CUSTOM
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

# =============================================================================
# ENHANCED DATA PROCESSOR
# =============================================================================
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
        print("\n[1/6] Loading and cleaning data...")
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
        print("\n[3/6] Aggregating to monthly data...")
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
        print("\n[4/6] Performing STL decomposition...")
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
        print("\n[5/6] Engineering features...")
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

    def calculate_adaptive_sequence_length(self, data_length):
        recommended_seq = max(6, min(12, data_length // 4))
        print(f"\n   [INFO] Data length: {data_length} months")
        print(f"   [INFO] Adaptive sequence length: {recommended_seq} months")
        return recommended_seq
    
    def prepare_sequences_multioutput(self, df, target_col, sequence_length, output_steps=12, test_size=0.2):
        print(f"\n[6/6] Preparing sequences...")
        try:
            self.feature_columns = [col for col in df.columns if col != target_col]
            print(f"   • Total features being used: {len(self.feature_columns)}")
            X = df[self.feature_columns].values
            y = df[target_col].values.reshape(-1, 1)
            self.target_min = y.min()
            self.target_max = y.max()
            X_scaled = self.scaler_features.fit_transform(X)
            y_scaled = self.scaler_target.fit_transform(y)
            X_seq, y_seq = [], []
            total_steps = sequence_length + output_steps
            if len(X_scaled) < total_steps:
                 raise ValueError(f"Data is too short. Need {total_steps} months, but only have {len(X_scaled)}")
            for i in range(len(X_scaled) - total_steps + 1):
                X_seq.append(X_scaled[i:i+sequence_length])
                y_seq.append(y_scaled[i+sequence_length:i+total_steps])
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            split_idx = int(len(X_seq) * (1 - test_size))
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            print(f"   ✓ Input/Output: {sequence_length}m / {output_steps}m")
            print(f"   ✓ Train/Test sequences: {len(X_train)} / {len(X_test)}")
            return X_train, X_test, y_train, y_test, df.index[sequence_length:]
        except Exception as e:
            print(f"   ⚠ Sequence preparation failed: {e}")
            raise

# =============================================================================
# ENHANCED MULTI-OUTPUT LSTM MODEL
# =============================================================================
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
            # Gunakan optimizer yang kompatibel dengan TF 2.13 lokal Anda
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
            variance_loss = BalancedVarianceMSE(variance_weight=0.05, seasonal_weight=0.1)
            
            model.compile(optimizer=optimizer, loss=variance_loss, metrics=['mae'])
            self.model = model
            print(f"   ✓ Model built successfully. Parameters: {model.count_params():,}")
            return model
        except Exception as e:
            print(f"   ⚠ Model building failed: {e}")
            raise
    
    def train(self, X_train, y_train, X_val, y_val, epochs=300, batch_size=4, verbose=1):
        print(f"\n[TRAINING] Starting training...")
        callbacks = [
            ModelCheckpoint('best_model_enhanced.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0),
            EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True, verbose=1, min_delta=1e-6),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-7, verbose=1, cooldown=10),
            TerminateOnNaN()
        ]
        self.history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks,
            verbose=verbose, shuffle=False
        )
        print("   ✓ Training completed!")
        return self.history
    
    def predict(self, X):
        try:
            predictions = self.model.predict(X, verbose=0)
            return np.maximum(predictions, 0)
        except Exception as e:
            print(f"   ⚠ Prediction failed: {e}")
            return np.zeros((X.shape[0], self.output_steps, 1))
    
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

# =============================================================================
# +++ PENYIMPANAN PROCESSOR +++
# =============================================================================
def save_processor_json(processor, sequence_length, timestamp, filepath_prefix='harvest_processor'):
    """
    Save processor parameters (scaler attributes, columns) as a JSON file.
    """
    print(f"\n[SAVING PROCESSOR PARAMETERS AS JSON]")
    try:
        os.makedirs('models', exist_ok=True)
        processor_path = f"models/{filepath_prefix}_params_{timestamp}.json" # <-- Berubah ke .json
        
        processor_data = {
            'scaler_features_center': processor.scaler_features.center_.tolist(),
            'scaler_features_scale': processor.scaler_features.scale_.tolist(),
            'scaler_target_min': processor.scaler_target.min_.tolist(),
            'scaler_target_scale': processor.scaler_target.scale_.tolist(),
            'feature_columns': processor.feature_columns,
            'sequence_length': sequence_length
        }
        
        with open(processor_path, 'w') as f:
            json.dump(processor_data, f, indent=4) 
        
        print(f"   ✓ Processor parameters (JSON) saved: {processor_path}")
        return processor_path
    except Exception as e:
        print(f"   ⚠ Save processor parameters failed: {e}")
        return None

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main(file_path='data_training.csv', forecast_months=12):
    """Enhanced training pipeline"""
    print("\n" + "="*80)
    print("STARTING ENHANCED TRAINING PIPELINE V10 (LOKAL)")
    print("="*80)
    
    try:
        processor = EnhancedDataProcessor()
        df_raw = processor.load_and_clean(file_path)
        df_clean = processor.create_datetime_index(df_raw)
        df_monthly = processor.aggregate_to_monthly(df_clean)
        processor.stl_decomposition(df_monthly, target_col='hasil_panen_kg')
        df_featured = processor.engineer_features(df_monthly, target_col='hasil_panen_kg')
        sequence_length = processor.calculate_adaptive_sequence_length(len(df_featured))
        
        X_train, X_test, y_train, y_test, dates = processor.prepare_sequences_multioutput(
            df_featured, 'hasil_panen_kg', sequence_length, forecast_months, 0.2
        )
        
        val_split = max(1, int(len(X_train) * 0.2))
        X_val, y_val = X_train[:val_split], y_train[:val_split]
        X_train_final, y_train_final = X_train[val_split:], y_train[val_split:]
        
        print(f"\n[DATA SUMMARY]")
        print(f"   Training/Validation/Test: {len(X_train_final)} / {len(X_val)} / {len(X_test)}")
        
        model_wrapper = EnhancedMultiOutputLSTM(sequence_length, X_train.shape[2], forecast_months)
        model_wrapper.build_model(lstm_units=64, dropout_rate=0.3)
        
        history = model_wrapper.train(
            X_train_final, y_train_final, X_val, y_val,
            epochs=300, batch_size=4, verbose=1
        )
        
        weights_path, timestamp = model_wrapper.save_model_weights('harvest_lstm_enhanced')
        processor_path = save_processor_json(processor, sequence_length, timestamp)
        
        print(f"\n[SUMMARY]")
        print(f"   Model weights saved: {weights_path}")
        print(f"   Processor params saved: {processor_path}")
        
        return { 'weights_path': weights_path, 'processor_path': processor_path }
        
    except Exception as e:
        print(f"\n[ERROR] Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# JALANKAN TRAINING
# =============================================================================
if __name__ == '__main__':
    results = main(
        file_path='data_training.csv',
        forecast_months=12
    )
    
    if results:
        print("\n" + "="*80)
        print("ENHANCED TRAINING V10 (LOKAL) COMPLETED SUCCESSFULLY")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("ENHANCED TRAINING V10 FAILED")
        print("="*80)
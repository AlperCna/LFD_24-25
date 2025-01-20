import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

class TFTModel:
    def __init__(self, file_path, target_column, features, lookback=24):
        """
        Temporal Fusion Transformer (TFT) model sınıfı.
        :param file_path: Veri setinin yolu
        :param target_column: Tahmin edilecek hedef sütun
        :param features: Özellik sütunlarının listesi
        :param lookback: Geçmiş veri uzunluğu
        """
        self.file_path = file_path
        self.target_column = target_column
        self.features = features
        self.lookback = lookback
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self):
        """
        Veriyi yükler ve ön işler.
        """
        data = pd.read_csv(self.file_path)
        data['Tarih'] = pd.to_datetime(data['Tarih'])
        data = data.sort_values(by='Tarih')
        data[self.features + [self.target_column]] = self.scaler.fit_transform(
            data[self.features + [self.target_column]]
        )
        return data

    def create_time_series(self, data, start_date, end_date):
        """
        Zaman serisi veri setini oluşturur.
        """
        data = data[(data['Tarih'] >= pd.to_datetime(start_date)) &
                    (data['Tarih'] <= pd.to_datetime(end_date))]
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[self.features].iloc[i:i + self.lookback].values)
            y.append(data[self.target_column].iloc[i + self.lookback])
        return np.array(X), np.array(y)

    def build_model(self):
        """
        TFT modelini oluşturur.
        """
        input_seq = Input(shape=(self.lookback, len(self.features)))
        lstm_out = LSTM(64, return_sequences=True)(input_seq)
        lstm_out = LSTM(64)(lstm_out)
        dense_out = Dense(64, activation='relu')(lstm_out)
        output = Dense(1, activation='linear')(dense_out)
        self.model = Model(inputs=input_seq, outputs=output)
        self.model.compile(optimizer='adam', loss='mse')
        return self.model

    def train_model(self, X_train, y_train, epochs=20, batch_size=32):
        """
        Modeli eğitir.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    def predict_and_evaluate(self, X_test, y_test):
        """
        Tahmin yapar ve performansı değerlendirir.
        """
        predictions = self.model.predict(X_test).flatten()
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return predictions, mae, rmse

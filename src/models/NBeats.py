import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model # type ignore
from tensorflow.keras.layers import Input, Dense, Flatten, Add, Reshape


class NBeatsModel:
    def __init__(self, file_path, target_column, lookback=10):
        """
        N-Beats model sınıfını başlatır.
        :param file_path: Veri setinin yolu
        :param target_column: Tahmin edilecek hedef sütun
        :param lookback: Zaman serisi için bakılacak önceki adım sayısı
        """
        self.file_path = file_path
        self.target_column = target_column
        self.lookback = lookback
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self):
        """
        Veriyi yükler ve ön işler.
        """
        data = pd.read_csv(self.file_path)
        data['Tarih'] = pd.to_datetime(data['Tarih'])
        data = data.sort_values(by='Tarih')
        self.data = data
        features = data[[self.target_column]].values
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled

    def create_time_series(self, data):
        """
        Zaman serisi veri setini oluşturur.
        """
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def build_model(self):
        """
        N-Beats modelini oluşturur.
        """
        input_layer = Input(shape=(self.lookback, 1))
        x = Flatten()(input_layer)

        # İlk Blok
        block_output = Dense(256, activation='relu')(x)
        block_output = Dense(self.lookback, activation='linear')(block_output)
        residual = Add()([block_output, x])

        # İkinci Blok
        block_output2 = Dense(256, activation='relu')(residual)
        block_output2 = Dense(self.lookback, activation='linear')(block_output2)

        # Çıkış Katmanı
        output_layer = Dense(1, activation='linear')(block_output2)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def train_model(self, X_train, y_train, epochs=20, batch_size=32):
        """
        Modeli eğitir.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    def predict_and_evaluate(self, X_test, y_test):
        """
        Test verisi üzerinde tahmin yapar ve sonuçları değerlendirir.
        """
        predictions = self.model.predict(X_test).flatten()
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return predictions, mae, rmse

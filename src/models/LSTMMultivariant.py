import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


class LSTMModelMultivariant:
    def __init__(self, file_path, target_column, features, look_back=24):

        self.file_path = file_path
        self.target_column = target_column
        self.features = features
        self.look_back = look_back
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self):
        """
        Veriyi yükler ve ön işler.
        """
        data = pd.read_csv(self.file_path)
        data['Tarih'] = pd.to_datetime(data['Tarih'])
        data = data.sort_values(by='Tarih')

        # Tüm özellikleri ve hedefi ölçekle
        all_columns = self.features + [self.target_column]
        data[all_columns] = self.scaler.fit_transform(data[all_columns])
        return data

    def create_sequences(self, data, look_back):
        """
        Zaman serisi veri setini LSTM için uygun sekanslara dönüştürür.
        """
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back, :-1])  # Özellikler
            y.append(data[i + look_back, -1])  # Hedef
        return np.array(X), np.array(y)

    def split_data(self, data, train_start_date, train_end_date, test_start_date, test_end_date):
        train_data = data[(data['Tarih'] >= pd.to_datetime(train_start_date)) &
                          (data['Tarih'] <= pd.to_datetime(train_end_date))]
        test_data = data[(data['Tarih'] >= pd.to_datetime(test_start_date)) &
                         (data['Tarih'] <= pd.to_datetime(test_end_date))]

        train_values = train_data[self.features + [self.target_column]].values
        test_values = test_data[self.features + [self.target_column]].values

        X_train, y_train = self.create_sequences(train_values, self.look_back)
        X_test, y_test = self.create_sequences(test_values, self.look_back)

        return X_train, y_train, X_test, y_test

    def build_model(self):
        """
        LSTM modelini oluşturur.
        """
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.look_back, len(self.features))),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.model = model
        return model

    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """
        LSTM modelini eğitir.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict_and_evaluate(self, X_test, y_test):
        """
        Tahmin yapar ve performansı değerlendirir.
        """
        predictions = self.model.predict(X_test).flatten()
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        return predictions, mae, rmse

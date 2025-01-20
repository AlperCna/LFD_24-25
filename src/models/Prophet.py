import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class ProphetModel:
    def __init__(self, file_path, target_column):
        """
        Prophet model sınıfını başlatır.
        :param file_path: Veri setinin yolu
        :param target_column: Tahmin edilecek hedef sütun
        """
        self.file_path = file_path
        self.target_column = target_column

    def load_and_preprocess_data(self):
        """
        Veriyi yükler ve Prophet formatına uygun hale getirir.
        """
        data = pd.read_csv(self.file_path)
        data['Tarih'] = pd.to_datetime(data['Tarih'])  # Tarih sütununu datetime formatına çevir
        data = data.rename(columns={'Tarih': 'ds', self.target_column: 'y'})  # Prophet için sütun isimleri
        return data

    def train_model(self, train_data):
        """
        Prophet modelini eğitir.
        :param train_data: Eğitim veri seti
        :return: Eğitilmiş model
        """
        model = Prophet()
        model.fit(train_data)
        self.model = model
        return model

    def predict_and_evaluate(self, test_data, period):
        """
        Tahmin yapar ve performansı değerlendirir.
        :param test_data: Test veri seti
        :param period: Tahmin dönemi (ör. 24 saat)
        :return: Tahminler, MAE, RMSE
        """
        future = self.model.make_future_dataframe(periods=period, freq='H')  # Saatlik tahminler
        forecast = self.model.predict(future)
        forecast = forecast[-len(test_data):]  # Sadece test dönemi için tahminler
        predictions = forecast['yhat'].values

        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        return predictions, mae, rmse

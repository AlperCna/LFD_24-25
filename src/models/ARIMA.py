import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class ARIMAModel:
    def __init__(self, file_path, target_column):
        """
        ARIMA model sınıfını başlatır.
        :param file_path: Veri setinin yolu
        :param target_column: Tahmin edilecek hedef sütun
        """
        self.file_path = file_path
        self.target_column = target_column

    def load_and_preprocess_data(self):
        """
        Veriyi yükler ve ön işler.
        """
        # Veriyi yükle
        data = pd.read_csv(self.file_path)
        data['Tarih'] = pd.to_datetime(data['Tarih'])  # Tarih sütununu datetime'a çevir
        data = data.sort_values(by='Tarih')  # Tarihe göre sıralama
        return data

    def train_model(self, train_data, p, d, q):
        """
        ARIMA modelini eğitir.
        :param train_data: Eğitim veri seti
        :param p: ARIMA modelindeki AR bileşeni (otomatik regresyon) için gecikme sayısı
        :param d: ARIMA modelindeki fark alma derecesi
        :param q: ARIMA modelindeki MA bileşeni (hareketli ortalama) gecikme sayısı
        :return: Eğitilmiş model
        """
        model = ARIMA(train_data, order=(p, d, q))
        self.model_fit = model.fit()
        print(self.model_fit.summary())
        return self.model_fit

    def predict_and_evaluate(self, test_data):
        """
        Tahmin yapar ve performansı değerlendirir.
        :param test_data: Test veri seti
        :return: Tahminler, MAE, RMSE
        """
        predictions = self.model_fit.forecast(steps=len(test_data))
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        return predictions, mae, rmse

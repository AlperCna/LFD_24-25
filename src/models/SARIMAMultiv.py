import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


class SARIMAModell:
    def __init__(self, file_path, target_column, features):
        """
        SARIMAX model sınıfını başlatır.
        :param file_path: Veri setinin yolu
        :param target_column: Tahmin edilecek hedef sütun
        :param features: Kullanılacak özelliklerin sütun adları
        """
        self.file_path = file_path
        self.target_column = target_column
        self.features = features

    def load_and_preprocess_data(self):
        """
        Veriyi yükler ve ön işler.
        """
        # Veriyi yükle
        data = pd.read_csv(self.file_path)
        data['Tarih'] = pd.to_datetime(data['Tarih'])  # Tarih sütununu datetime'a çevir
        data = data.sort_values(by='Tarih')  # Tarihe göre sıralama
        return data

    def train_model(self, train_data, exogenous_train, p, d, q, P, D, Q, s):
        """
        SARIMAX modelini eğitir.
        :param train_data: Eğitim veri seti
        :param exogenous_train: Eksojen değişkenler (bağımsız değişkenler) için eğitim veri seti
        :param p: ARIMA modelindeki AR bileşeni için gecikme sayısı
        :param d: ARIMA modelindeki fark alma derecesi
        :param q: ARIMA modelindeki MA bileşeni için gecikme sayısı
        :param P, D, Q, s: Mevsimsel bileşenlerin parametreleri ve mevsimsel dönem uzunluğu
        :return: Eğitilmiş model
        """
        model = SARIMAX(
            train_data,
            exog=exogenous_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fit = model.fit(disp=False)
        print(self.model_fit.summary())
        return self.model_fit

    def predict_and_evaluate(self, test_data, exogenous_test):
        """
        Tahmin yapar ve performansı değerlendirir.
        :param test_data: Test veri seti
        :param exogenous_test: Eksojen değişkenler için test veri seti
        :return: Tahminler, MAE, RMSE
        """
        predictions = self.model_fit.forecast(steps=len(test_data), exog=exogenous_test)
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        return predictions, mae, rmse

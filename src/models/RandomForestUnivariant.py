import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RandomForestModel:
    def __init__(self, file_path, target_column, random_state=42):
        """
        Random Forest model sınıfını başlatır.
        :param file_path: Veri setinin yolu
        :param target_column: Tahmin edilecek hedef sütun
        :param random_state: Rastgelelik kontrolü için seed
        """
        self.file_path = file_path
        self.target_column = target_column
        self.random_state = random_state

    def load_and_preprocess_data(self):
        """
        Veriyi yükler ve ön işler.
        """
        # Veriyi yükle
        data = pd.read_csv(self.file_path)
        data['Tarih'] = pd.to_datetime(data['Tarih'])  # Tarih sütununu datetime'a çevir
        data = data.sort_values(by='Tarih')  # Tarihe göre sıralama
        return data

    def split_data(self, data, train_end_date, test_start_date, test_end_date):
        """
        Eğitim ve test verisini tarih bazlı ayırır.
        :param data: Veri seti
        :param train_end_date: Eğitim verisinin bitiş tarihi
        :param test_start_date: Test verisinin başlangıç tarihi
        :param test_end_date: Test verisinin bitiş tarihi
        :return: X_train, X_test, y_train, y_test
        """
        train_data = data.loc[:train_end_date]
        test_data = data.loc[test_start_date:test_end_date]

        X_train = train_data.drop(columns=["Tarih", self.target_column])
        y_train = train_data[self.target_column]
        X_test = test_data.drop(columns=["Tarih", self.target_column])
        y_test = test_data[self.target_column]

        # Eksik ve sonsuz değerleri temizle
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Modeli eğitir.
        """
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state)
        model.fit(X_train, y_train)
        self.model = model
        print("Model başarıyla eğitildi!")
        return model

    def predict_and_evaluate(self, X_test, y_test):
        """
        Tahmin yapar ve performansı değerlendirir.
        """
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        return y_pred, mae, rmse

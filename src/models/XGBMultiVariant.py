import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


class XGBoostModelMultivaryant:
    def __init__(self, file_path, target_column, features, random_state=42):
        """
        XGBoost multivaryant model sınıfını başlatır.
        :param file_path: Veri setinin yolu
        :param target_column: Tahmin edilecek hedef sütun
        :param features: Kullanılacak özellikler listesi
        :param random_state: Rastgelelik kontrolü için seed
        """
        self.file_path = file_path
        self.target_column = target_column
        self.features = features
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def load_and_preprocess_data(self):
        """
        Veriyi yükler ve ön işler.
        """
        data = pd.read_csv(self.file_path)
        data['Tarih'] = pd.to_datetime(data['Tarih'])
        data = data.sort_values(by='Tarih')
        selected_columns = ['Tarih', self.target_column] + self.features
        data = data[selected_columns]
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
        train_data = data[data['Tarih'] <= pd.to_datetime(train_end_date)]
        test_data = data[(data['Tarih'] >= pd.to_datetime(test_start_date)) & (data['Tarih'] <= pd.to_datetime(test_end_date))]
        X_train = train_data[self.features]
        y_train = train_data[self.target_column]
        X_test = test_data[self.features]
        y_test = test_data[self.target_column]

        # Eksik ve sonsuz değerleri temizle
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """
        Varsayılan hiperparametrelerle modeli eğitir.
        """
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=self.random_state)
        model.fit(X_train, y_train)
        self.model = model
        print("Model başarıyla eğitildi!")
        return model

    def train_model_with_grid_search(self, X_train, y_train):
        """
        GridSearchCV kullanarak hiperparametre optimizasyonu yapar ve modeli eğitir.
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 10, 100],
        }

        xgb_model = xgb.XGBRegressor(random_state=self.random_state)

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # En iyi parametreleri kaydet
        self.best_params = grid_search.best_params_
        print("En iyi hiperparametreler:", self.best_params)

        # En iyi modelle eğit
        self.model = grid_search.best_estimator_
        print("Model başarıyla eğitildi!")
        return self.model

    def predict_and_evaluate(self, X_test, y_test):
        """
        Tahmin yapar ve performansı değerlendirir.
        """
        if self.model is None:
            raise ValueError("Model eğitilmedi.")
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        return y_pred, mae, rmse

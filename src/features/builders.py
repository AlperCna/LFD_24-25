import pandas as pd

# 1. Zamana Dayalı Özellikler
def add_time_features(df):

    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df['Year'] = df['Tarih'].dt.year
    df['Month'] = df['Tarih'].dt.month
    df['Day'] = df['Tarih'].dt.day
    df['Hour'] = df['Tarih'].dt.hour

    # Mevsim ekleme
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['Season'] = df['Month'].apply(get_season)
    return df

# 2. Hareketli İstatistikler
def add_rolling_statistics(df, column, window=3):

    df[f'{column}_rolling_mean'] = df[column].rolling(window=window).mean()
    df[f'{column}_rolling_sum'] = df[column].rolling(window=window).sum()
    df[f'{column}_rolling_std'] = df[column].rolling(window=window).std()
    return df

# 3. Hava Durumundan Türetilecek Özellikler
def add_weather_features(df):

    # İstanbul
    df['Istanbul_temp_feels_like'] = df['Istanbul_temp'] - (df['Istanbul_rhum'] * 0.1)
    df['Istanbul_wind_chill'] = df['Istanbul_wspd'] * 0.7

    # İzmir
    df['Izmir_temp_feels_like'] = df['Izmir_temp'] - (df['Izmir_rhum'] * 0.1)
    df['Izmir_wind_chill'] = df['Izmir_wspd'] * 0.7

    # Samsun
    df['Samsun_temp_feels_like'] = df['Samsun_temp'] - (df['Samsun_rhum'] * 0.1)
    df['Samsun_wind_chill'] = df['Samsun_wspd'] * 0.7

    # Antalya
    df['Antalya_temp_feels_like'] = df['Antalya_temp'] - (df['Antalya_rhum'] * 0.1)
    df['Antalya_wind_chill'] = df['Antalya_wspd'] * 0.7

    # Sivas
    df['Sivas_temp_feels_like'] = df['Sivas_temp'] - (df['Sivas_rhum'] * 0.1)
    df['Sivas_wind_chill'] = df['Sivas_wspd'] * 0.7

    # Diyarbakır
    df['Diyarbakır_temp_feels_like'] = df['Diyarbakır_temp'] - (df['Diyarbakır_rhum'] * 0.1)
    df['Diyarbakır_wind_chill'] = df['Diyarbakır_wspd'] * 0.7

    # Erzurum
    df['Erzurum_temp_feels_like'] = df['Erzurum_temp'] - (df['Erzurum_rhum'] * 0.1)
    df['Erzurum_wind_chill'] = df['Erzurum_wspd'] * 0.7

    return df


# 4. Ekonomik Göstergeler
def add_economic_indicators(df):

    df['price_change'] = df['Smf'].pct_change()  # Fiyat yüzdesel değişim
    df['demand_supply_ratio'] = df['Talepislemhacmi'] / df['Arzislemhacmi']  # Talep/Arz oranı
    df['price_volatility'] = df['Smf'].rolling(window=7).std()  # Fiyat volatilitesi
    return df

# 5. Tüm Özellikleri Uygulama
def apply_feature_engineering(df):
    """
    Tüm özellik mühendisliği adımlarını uygular ve sonuçları yuvarlar.

    Args:
        df (pd.DataFrame): Veri çerçevesi.

    Returns:
        pd.DataFrame: Özellik mühendisliği tamamlanmış ve yuvarlanmış veri çerçevesi.
    """
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_economic_indicators(df)

    # Rolling statistics yalnızca belirli sütunlar için eklenir
    for column in ['Smfdolar', 'Ptfdolar', 'Talepislemhacmi']:
        df = add_rolling_statistics(df, column=column, window=3)

    # Tüm sayısal değerleri 2 basamağa yuvarla
    df = df.round(2)
    return df


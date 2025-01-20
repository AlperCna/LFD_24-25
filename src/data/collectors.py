# from meteostat import Point, Hourly
# from datetime import datetime
# import pandas as pd
#
# def get_hourly_weather_data(cities, start_date, end_date):
#     """
#     Belirtilen şehirler ve tarih aralığı için saatlik hava durumu verilerini çeker ve birleştirir.
#
#     Args:
#         cities (dict): Şehirlerin isimlerini ve koordinatlarını içeren bir sözlük.
#                        Örn: {"Istanbul": (41.0082, 28.9784)}
#         start_date (str): Başlangıç tarihi (YYYY-MM-DD formatında).
#         end_date (str): Bitiş tarihi (YYYY-MM-DD formatında).
#
#     Returns:
#         pd.DataFrame: Tüm şehirlerin birleşik hava durumu verilerini içeren DataFrame.
#     """
#     # Tarih aralığını datetime formatına çevir
#     start = datetime.strptime(start_date, '%Y-%m-%d')
#     end = datetime.strptime(end_date, '%Y-%m-%d')
#
#     # Tüm şehirlerden alınan verileri tutacak bir liste
#     all_city_data = []
#
#     for city_name, coords in cities.items():
#         # Şehir için Point oluştur
#         location = Point(coords[0], coords[1])
#
#         # Şehirden saatlik veri çek
#         datas = Hourly(location, start, end)
#         datas = datas.fetch()
#
#         # Eksik sütunları kontrol et ve sadece sıcaklık, nem ve yağış verilerini seç
#         if 'temp' not in datas.columns:
#             datas['temp'] = None
#         if 'rhum' not in datas.columns:
#             datas['rhum'] = None
#         if 'prcp' not in datas.columns:
#             datas['prcp'] = None
#
#         # İlgili sütunları seç ve şehir adını sütun isimlerine ekle
#         datas = datas[['temp', 'rhum', 'prcp']].rename(columns={
#             'temp': f"{city_name}_temp",
#             'rhum': f"{city_name}_rhum",
#             'prcp': f"{city_name}_prcp"
#         })
#
#         # Tarih sütununu ekle ve datetime formatına çevir
#         datas = datas.reset_index()
#         datas['time'] = pd.to_datetime(datas['time'])
#
#         # Şehir verilerini listeye ekle
#         all_city_data.append(datas.set_index('time'))
#
#     # Tüm şehir verilerini pd.concat ile birleştir
#     combined_data = pd.concat(all_city_data, axis=1)
#
#     # Eğer 'time' indeksi varsa ve sütun değilse, sıfırlayıp sütuna çevir
#     if combined_data.index.name == 'time':
#         combined_data = combined_data.reset_index()
#
#     # İndeksi datetime olarak kontrol et ve adlandır
#     if 'time' in combined_data.columns:
#         combined_data = combined_data.rename(columns={'time': 'datetime'})
#
#     # Eğer datetime sütunu yoksa, hata oluşabilir. Kontrol ekleyelim
#     if 'datetime' not in combined_data.columns:
#         raise ValueError("Birleştirme işlemi sırasında 'datetime' sütunu oluşturulamadı!")
#
#     # Tüm datetime sütunlarının doğru formatta olduğundan emin olun
#     combined_data['datetime'] = pd.to_datetime(combined_data['datetime'])
#
#     return combined_data



from meteostat import Point, Hourly
from datetime import datetime
import pandas as pd

def get_hourly_weather_data(cities, start_date, end_date):

    # Tarih aralığını datetime formatına çevir
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Tüm şehirlerden alınan verileri tutacak bir liste
    all_city_data = []

    for city_name, coords in cities.items():
        # Şehir için Point oluştur
        location = Point(coords[0], coords[1])

        # Şehirden saatlik veri çek
        data = Hourly(location, start, end)
        data = data.fetch()

        # Eksik sütunları kontrol et ve sadece sıcaklık, nem ve rüzgar verilerini seç
        if 'temp' not in data.columns:
            data['temp'] = None
        if 'rhum' not in data.columns:
            data['rhum'] = None
        if 'wspd' not in data.columns:
            data['wspd'] = None
        if 'wdir' not in data.columns:
            data['wdir'] = None

        # İlgili sütunları seç ve şehir adını sütun isimlerine ekle
        data = data[['temp', 'rhum', 'wspd', 'wdir']].rename(columns={
            'temp': f"{city_name}_temp",
            'rhum': f"{city_name}_rhum",
            'wspd': f"{city_name}_wspd",
            'wdir': f"{city_name}_wdir"
        })

        # Tarih sütununu ekle ve datetime formatına çevir
        data = data.reset_index()
        data['time'] = pd.to_datetime(data['time'])

        # Şehir verilerini listeye ekle
        all_city_data.append(data.set_index('time'))

    # Tüm şehir verilerini pd.concat ile birleştir
    combined_data = pd.concat(all_city_data, axis=1)

    # Eğer 'time' indeksi varsa ve sütun değilse, sıfırlayıp sütuna çevir
    if combined_data.index.name == 'time':
        combined_data = combined_data.reset_index()

    # İndeksi datetime olarak kontrol et ve adlandır
    if 'time' in combined_data.columns:
        combined_data = combined_data.rename(columns={'time': 'datetime'})

    # Eğer datetime sütunu yoksa, hata oluşabilir. Kontrol ekleyelim
    if 'datetime' not in combined_data.columns:
        raise ValueError("Birleştirme işlemi sırasında 'datetime' sütunu oluşturulamadı!")

    # Tüm datetime sütunlarının doğru formatta olduğundan emin olun
    combined_data['datetime'] = pd.to_datetime(combined_data['datetime'])

    return combined_data





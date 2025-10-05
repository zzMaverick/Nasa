# -*- coding: utf-8 -*-
import xarray as xr
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# ==============================
# CONFIGURAÇÕES INICIAIS
# ==============================

data_path = "Goiania2009-2019/*.nc4"  # pasta com dados TRMM
lat_min, lat_max = -17.1869, -16.1869
lon_min, lon_max = -49.7648, -48.7648

# ==============================
# LISTA DE ALAGAMENTOS REAIS (Marginal Botafogo)
# ==============================
# Dias obtidos de jornais locais (G1, MaisGoiás, O Popular, 2009–2019)
alagamentos_reais = [
    "2010-02-14", "2010-12-28",
    "2013-03-07", "2013-12-21",
    "2014-11-25", "2014-12-05",
    "2015-12-26", "2015-12-30",
    "2016-01-10", "2016-12-03",
    "2017-02-17", "2017-03-05",
    "2018-01-10", "2018-12-06",
    "2019-01-12", "2019-11-27"
]
alagamentos_reais = [pd.to_datetime(d).date() for d in alagamentos_reais]

# ==============================
# FUNÇÃO PARA DEFINIR QUADRANTE
# ==============================
lat_center = (lat_min + lat_max) / 2
lon_center = (lon_min + lon_max) / 2
delta = 0.25  # tamanho de cada zona

def define_quadrante(lat, lon):
    if lat < lat_center - delta:
        lat_label = 'S'
    elif lat > lat_center + delta:
        lat_label = 'N'
    else:
        lat_label = 'C'

    if lon < lon_center - delta:
        lon_label = 'W'
    elif lon > lon_center + delta:
        lon_label = 'E'
    else:
        lon_label = 'C'

    return lat_label + lon_label  # ex: "NW", "CC", "SE"

# ==============================
# LEITURA E PROCESSAMENTO DOS ARQUIVOS
# ==============================
all_data = []

for file in sorted(glob.glob(data_path)):
    ds = xr.open_dataset(file)
    precip = ds['precipitation']
    precip_goiania = precip.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    date_str = file.split(".")[1]
    date = datetime.strptime(date_str, "%Y%m%d").date()

    lats = precip_goiania['lat'].values
    lons = precip_goiania['lon'].values
    precip_values = precip_goiania.values
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

    df = pd.DataFrame({
        'date': date,
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten(),
        'precip_mm': precip_values.flatten()
    })

    df['quadrante'] = [define_quadrante(la, lo) for la, lo in zip(df.lat, df.lon)]
    all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)

# ==============================
# ANÁLISE DE ANOMALIA
# ==============================
stats = full_df.groupby(['lat', 'lon'])['precip_mm'].agg(['mean', 'std']).reset_index()
full_df = full_df.merge(stats, on=['lat', 'lon'])
full_df['anomalia'] = (full_df['precip_mm'] - full_df['mean']) / full_df['std']

# ==============================
# AGREGAÇÃO POR DIA E QUADRANTE
# ==============================
agg = full_df.groupby(['date', 'quadrante'])['anomalia'].mean().unstack().fillna(0)
agg.reset_index(inplace=True)

# ==============================
# ADICIONAR COLUNA DE ALAGAMENTO (1 = sim, 0 = não)
# ==============================
agg['alagou'] = agg['date'].apply(lambda x: 1 if x in alagamentos_reais else 0)

# ==============================
# MODELO DE REGRESSÃO
# ==============================
X = agg.drop(columns=['date', 'alagou'])
y = agg['alagou']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = (model.predict(X_test) > 0.5).astype(int)

# ==============================
# RELATÓRIO DE DESEMPENHO
# ==============================
print("\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_test, y_pred))

# ==============================
# IMPORTÂNCIA DOS QUADRANTES
# ==============================
coef_df = pd.DataFrame({
    'Quadrante': X.columns,
    'Peso': model.coef_
}).sort_values(by='Peso', ascending=False)

print("\n💧 IMPORTÂNCIA DOS QUADRANTES:")
print(coef_df)

plt.figure(figsize=(8, 5))
sns.barplot(x='Peso', y='Quadrante', data=coef_df, hue='Quadrante', palette='coolwarm', legend=False)
plt.title("Importância dos Quadrantes na Probabilidade de Alagamento")
plt.xlabel("Peso (Influência)")
plt.ylabel("Quadrante")
plt.show()

# ==============================
# MAPA DE CALOR DAS ANOMALIAS
# ==============================
anomalia_media = full_df.groupby(['lat', 'lon'])['anomalia'].mean().unstack()
plt.figure(figsize=(10, 6))
sns.heatmap(anomalia_media, cmap='coolwarm', center=0)
plt.title("Mapa de Calor - Anomalias Médias de Precipitação (2009-2019)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

print("\n✅ Modelo executado com sucesso!")

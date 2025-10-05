import pandas as pd
import folium
from folium.plugins import HeatMap
import numpy as np

# Ler CSV com precipitação e anomalias
df = pd.read_csv("precipitacao_goiania_2009_2019.csv")

# Normalizar a anomalia para visualização (0 a 1)
anom_min = df['anomalia'].min()
anom_max = df['anomalia'].max()
df['anomalia_norm'] = (df['anomalia'] - anom_min) / (anom_max - anom_min)

# Criar mapa centrado em Goiânia
mapa = folium.Map(location=[-16.6869, -49.2648], zoom_start=11, tiles="CartoDB positron")

# Criar lista de dados para HeatMap: [lat, lon, intensidade]
heat_data = [[row['lat'], row['lon'], row['anomalia_norm']] for idx, row in df.iterrows()]

# Adicionar HeatMap com parâmetros que deixam o mapa mais claro e suave
HeatMap(
    heat_data,
    radius=20,      # aumenta o "tamanho" de cada ponto
    blur=15,        # suaviza transição entre pontos
    min_opacity=0.3 # pontos claros ainda aparecem
).add_to(mapa)

# Adicionar marcador da Marginal Botafogo
folium.Marker(
    location=[-16.6869, -49.2648],
    popup="Marginal Botafogo",
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(mapa)

# Salvar mapa interativo
mapa.save("mapa_calor_precipitacao_goiania_completo.html")
print("Mapa interativo criado: mapa_calor_precipitacao_goiania_completo.html")

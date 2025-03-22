import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Parámetros
simbolo = "BTCUSDT"
intervalo = "1h"  
limite = 500  

# URL de la API de velas
url_klines = f"https://api.binance.com/api/v3/klines?symbol={simbolo}&interval={intervalo}&limit={limite}"

# Obtener datos de velas
response = requests.get(url_klines)
data = response.json()

# Convertir datos a DataFrame
df = pd.DataFrame(data, columns=[
    "timestamp", "apertura", "máximo", "mínimo", "cierre", "volumen", 
    "tiempo_cierre", "volumen_activo", "numero_de_transacciones", 
    "volumen_comprador_base", "volumen_comprador_cotizacion", "ignorar"
])

# Convertir valores numéricos
df = df.astype({
    "apertura": float, "máximo": float, "mínimo": float, "cierre": float, "volumen": float,
    "volumen_activo": float, "numero_de_transacciones": int,
    "volumen_comprador_base": float, "volumen_comprador_cotizacion": float
})

# Convertir timestamp a fecha
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("timestamp", inplace=True)

# Obtener datos de trades recientes
url_trades = f"https://api.binance.com/api/v3/trades?symbol={simbolo}&limit=500"
response_trades = requests.get(url_trades)
data_trades = response_trades.json()
df_trades = pd.DataFrame(data_trades)
df_trades["time"] = pd.to_datetime(df_trades["time"], unit="ms")
df_trades.set_index("time", inplace=True)

# Análisis descriptivo
print(df.head())
print(df.info())
print(df.describe())

# Limpieza de datos
print(df.isnull().sum())
df_clean = df.dropna()
print(df_clean.duplicated().sum())
df_clean = df_clean.drop_duplicates()

# Análisis de volumen inusual
df_clean["volumen_variacion"] = df_clean["volumen"].pct_change()

# Detectar variaciones extremas de volumen
z_scores = np.abs((df_clean["volumen_variacion"] - df_clean["volumen_variacion"].mean()) / df_clean["volumen_variacion"].std())
outliers = df_clean[z_scores > 3]
print("Eventos con variaciones extremas de volumen:")
print(outliers)

sns.histplot(df_clean["volumen_variacion"], bins=50, kde=True)
plt.title("Distribución de variación de volumen")
plt.show()

# Análisis de transacciones frecuentes en trades
df_trades["frecuencia"] = df_trades.index.value_counts()

sns.histplot(df_trades["frecuencia"], bins=50, kde=True)
plt.title("Distribución de frecuencia de trades")
plt.show()

# Matriz de correlación
matriz_correlacion = df_clean.corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de correlación')
plt.show()

# Análisis de la matriz de correlación
correlaciones = matriz_correlacion[(matriz_correlacion > 0.7) | (matriz_correlacion < -0.7)]
print("\nVariables con fuerte correlación:")
print(correlaciones)

# Detección de actividad sospechosa en órdenes
url_depth = f"https://api.binance.com/api/v3/depth?symbol={simbolo}&limit=100"
response_depth = requests.get(url_depth)
data_depth = response_depth.json()

df_bids = pd.DataFrame(data_depth['bids'], columns=['precio', 'cantidad'])
df_asks = pd.DataFrame(data_depth['asks'], columns=['precio', 'cantidad'])

df_bids['cantidad'] = df_bids['cantidad'].astype(float)
df_asks['cantidad'] = df_asks['cantidad'].astype(float)

sns.histplot(df_bids['cantidad'], bins=50, kde=True, color='blue', label='Compras')
sns.histplot(df_asks['cantidad'], bins=50, kde=True, color='red', label='Ventas')
plt.legend()
plt.title("Distribución de órdenes en el libro de órdenes")
plt.show()

# Detección de Pump & Dump
df_clean['precio_cambio'] = df_clean['cierre'].pct_change()
df_clean['volumen_cambio'] = df_clean['volumen'].pct_change()

pump_and_dump = df_clean[(df_clean['precio_cambio'] > 0.1) & (df_clean['volumen_cambio'] > 0.5)]
print("Eventos sospechosos de Pump & Dump:")
print(pump_and_dump)

# Relación entre volumen y cambio de precio
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_clean, x='volumen_variacion', y='precio_cambio', alpha=0.5)
plt.title("Relación entre variación de volumen y cambio de precio")
plt.xlabel("Variación del Volumen")
plt.ylabel("Cambio del Precio")
plt.show()

# Superposición de variaciones en el tiempo
fig, ax1 = plt.subplots(figsize=(10,6))

ax2 = ax1.twinx()
ax1.plot(df_clean.index, df_clean['volumen_variacion'], color='blue', label="Variación de Volumen", alpha=0.6)
ax2.plot(df_clean.index, df_clean['precio_cambio'], color='red', label="Cambio de Precio", alpha=0.6)

ax1.set_xlabel("Tiempo")
ax1.set_ylabel("Variación de Volumen", color='blue')
ax2.set_ylabel("Cambio de Precio", color='red')

fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
plt.title("Comparación de Variación de Volumen y Cambio de Precio en el Tiempo")
plt.show()

# Matriz de correlación entre volumen y precio
correlacion = df_clean[['volumen_variacion', 'precio_cambio']].corr()
print("Correlación entre variación de volumen y cambio de precio:")
print(correlacion)


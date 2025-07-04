# este script es para crear las nuevas variables que se van a usar en el modelo de prediccion

import pandas as pd

# Carga el archivo unificado
df = pd.read_csv('./data/cryptos_unificado.csv')

# Aseguro que Date sea tipo datetime
df['Date'] = pd.to_datetime(df['Date'])

# Ordeno por Name y Date
df = df.sort_values(by=['Name', 'Date']).reset_index(drop=True)

# == Variables nuevas == #
# Retorno diario
df['daily_return'] = (df['Close'] - df['Open']) / df['Open']

# Volatilidad diaria
df['volatility'] = (df['High'] - df['Low']) / df['Open']

# Cambio porcentual del precio a 7 días
df['price_change_7d'] = df.groupby('Name')['Close'].pct_change(periods=7)

# Cambio porcentual del volumen día a día
df['volume_change_1d'] = df.groupby('Name')['Volume'].pct_change(periods=1)

# Capitalización promedio por criptomoneda
avg_marketcap = df.groupby('Name')['Marketcap'].mean().sort_values(ascending=False)
rank_dict = {name: rank + 1 for rank, name in enumerate(avg_marketcap.index)}

# Ranking de capitalización
df['marketcap_rank'] = df['Name'].map(rank_dict)

# Reordenar columnas para mayor claridad
cols = ['Name', 'Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap',
        'marketcap_rank', 'daily_return', 'volatility', 'price_change_7d', 'volume_change_1d']
df = df[cols]

# Guardamos el nuevo dataset enriquecido
df.to_csv('./data/cryptos_enriquecido.csv', index=False)

print("Dataset enriquecido guardado como 'cryptos_enriquecido.csv'")
print(df.head())

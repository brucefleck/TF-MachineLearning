import pandas as pd

# Cargar el dataset enriquecido
df = pd.read_csv('./data/cryptos_enriquecido.csv')

# Asegurar formato de fecha
df['Date'] = pd.to_datetime(df['Date'])

# Ordenar por criptomoneda y fecha
df = df.sort_values(by=['Name', 'Date']).reset_index(drop=True)

# Crear columna de precio futuro (Close 30 días adelante)
df['Close_30d_ahead'] = df.groupby('Name')['Close'].shift(-30)

# Calcular el crecimiento porcentual en 30 días
df['future_return_30d'] = (df['Close_30d_ahead'] - df['Close']) / df['Close']

# Definir variable objetivo: 1 si crece más de 30%, si no 0
df['target_growth_30d'] = df['future_return_30d'].apply(lambda x: 1 if x is not None and x > 0.3 else 0)

# Elimina filas donde no se puede calcular el target (últimos 30 días por cripto)
df = df.dropna(subset=['Close_30d_ahead'])

# Guardar nuevo dataset con target
df.to_csv('./data/data_final.csv', index=False)

print("Variable objetivo 'target_growth_30d' creada y guardada en 'data_final.csv'")
print(df[['Name', 'Date', 'Close', 'Close_30d_ahead', 'future_return_30d', 'target_growth_30d']].head())

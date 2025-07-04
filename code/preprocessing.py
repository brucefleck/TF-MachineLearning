import pandas as pd
import os

folder_path = './data'

dataframes = []

# recorre cada archivo en la carpeta original del dataset y junta todos los datos en un solo dataframe 
for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and filename.startswith('coin_'):        
        # Carga el CSV
        df = pd.read_csv(os.path.join(folder_path, filename))
        
        # Convertimos la fecha a formato datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Asegura el orden por fecha
        df = df.sort_values(by='Date')
        
        # Agregamos el DataFrame a la lista
        dataframes.append(df)

# Concatena todos los DataFrames en uno solo
full_df = pd.concat(dataframes, ignore_index=True)

# Reordena las columnas
cols = ['Name', 'Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap']
full_df = full_df[cols]

# Mostramos un resumen
print("Dataset unificado con forma:", full_df.shape)
print(full_df.head())

# finalmente, guarda el dataframe unificado en un nuevo archivo CSV
full_df.to_csv('cryptos_unificado.csv', index=False)
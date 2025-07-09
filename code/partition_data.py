import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv('../data/data_completo.csv')

# Variables predictoras
features = ['daily_return', 'volatility', 'price_change_7d', 'volume_change_1d', 'marketcap_rank']
X = df[features]
y = df['target_growth_30d']

# Divide en train/test
# 1. Primero dividimos en train y temp (train = 70%, temp = 30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 2. Luego dividimos temp en validation y test (15% y 15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Confirmacion de tamaños
print(f"Train: {X_train.shape[0]} muestras")
print(f"Validation: {X_val.shape[0]} muestras")
print(f"Test: {X_test.shape[0]} muestras")

# Guardar los conjuntos de datos
X_train.to_csv('../data/X_train.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)

X_val.to_csv('../data/X_val.csv', index=False)
y_val.to_csv('../data/y_val.csv', index=False)

X_test.to_csv('../data/X_test.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)

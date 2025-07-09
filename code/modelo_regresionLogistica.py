import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Cargar dataset
df = pd.read_csv('../data/data_completo.csv')

# Variables predictoras
features = ['daily_return', 'volatility', 'price_change_7d', 'volume_change_1d', 'marketcap_rank']
X = df[features]
y = df['target_growth_30d']

# Divide en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Modelo base
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
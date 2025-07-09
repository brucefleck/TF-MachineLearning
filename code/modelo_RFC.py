import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# MODELO RANDOM FOREST CLASSIFIER

# cargar el dataset para entrenar y validar el modelo
X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv')

X_val = pd.read_csv('../data/X_val.csv')
y_val = pd.read_csv('../data/y_val.csv')

# Entrenar el modelo
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Predicciones
y_pred = rf.predict(X_val)

# Evaluación
print("MODELO RANDOM FOREST CLASSIFIER")
print("Matriz de Confusión:")
print(confusion_matrix(y_val, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_val, y_pred))

# Importancia de características
features = ['daily_return', 'volatility', 'price_change_7d', 'volume_change_1d', 'marketcap_rank']
importances = rf.feature_importances_
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.4f}")

# guardarmos el modelo
joblib.dump(rf, '../modelos/modelo_RFC.pkl')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# MODELO REGRESION LOGISTICA

# cargar el dataset para entrenar y validar el modelo
X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv')

X_val = pd.read_csv('../data/X_val.csv')
y_val = pd.read_csv('../data/y_val.csv')

# Modelo base
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_val)

# Evaluación
print("MODELO REGRESION LOGISTICA")
print("Matriz de Confusión:")
print(confusion_matrix(y_val, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_val, y_pred))

# guardarmos el modelo
joblib.dump(model, '../modelos/modelo_RL.pkl')
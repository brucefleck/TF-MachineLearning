import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# MODELO MLP (Multi Layer Perceptron)

# cargar el dataset para entrenar y validar el modelo
X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv')

X_val = pd.read_csv('../data/X_val.csv')
y_val = pd.read_csv('../data/y_val.csv')

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_val)

# Crear y entrenar el modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Evaluación
y_pred = mlp.predict(X_test_scaled)
print("MODELO MLP")
print("Matriz de Confusión:")
print(confusion_matrix(y_val, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_val, y_pred))

# guardarmos el modelo
joblib.dump(mlp, '../modelos/modelo_MLP.pkl')
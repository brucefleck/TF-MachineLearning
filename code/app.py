import streamlit as st
import joblib
import numpy as np

st.title("Predicción de Crecimiento de Criptomonedas")
st.subheader("Selecciona un modelo y proporciona las características")

# === Selección de modelo ===
model_option = st.selectbox("Selecciona el modelo:", 
                            ["Random Forest", "Regresión Logística", "Red Neuronal (MLP)"])

# Cargar el modelo seleccionado
if model_option == "Random Forest":
    model = joblib.load('../modelos/modelo_RFC.pkl')
    scaler = None
elif model_option == "Regresión Logística":
    model = joblib.load('../modelos/modelo_RL.pkl')
    scaler = None
elif model_option == "Red Neuronal (MLP)":
    model = joblib.load('../modelos/modelo_MLP.pkl')

# === Entradas del usuario ===
st.markdown("### Ingrese los datos de la criptomoneda:")

daily_return = st.number_input("Retorno diario", value=0.01, step=0.001)
volatility = st.number_input("Volatilidad relativa", value=0.02, step=0.001)
price_change_7d = st.number_input("Cambio porcentual del precio a 7 días", value=0.05, step=0.001)
volume_change_1d = st.number_input("Variación diaria del volumen", value=0.10, step=0.001)
marketcap_rank = st.number_input("Ranking promedio de capitalización", min_value=1.0, max_value=500.0, value=100.0, step=0.001)

# === Botón de predicción ===
if st.button("Predecir"):
    input_data = np.array([[daily_return, volatility, price_change_7d, volume_change_1d, marketcap_rank]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Mostrar resultado
    if prediction == 1:
        st.success(f"¡Probabilidad de crecimiento alto: {prob:.2%}!")
    else:
        st.warning(f"Probabilidad baja de crecimiento: {prob:.2%}")

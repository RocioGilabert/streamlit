import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Función para cargar el modelo entrenado desde un archivo PKL
@st.cache_resource
def load_model():
    
    model_filename = "prediccion_viajeros.pkl"
    with open(model_filename, "rb") as model_file:
        model = pickle.load(model_file)
    return model

@st.cache_data
def load_province_data():
    
    df = pd.read_pickle("DATA/final_data.pkl")
    province_mapping = df.drop_duplicates().set_index("provincia_origen_name")["provincia_origen"].to_dict()
    return province_mapping


@st.cache_data
def load_weather_data():
    
    weather_data = pd.read_pickle("DATA/numeric_data.pkl")
    return weather_data

# Función para obtener las últimas condiciones meteorológicas para una provincia destino
def get_last_weather_conditions(provincia_destino_codigo):
    
    last_weather = weather_data[weather_data["provincia_destino"] == provincia_destino_codigo].sort_values("day_of_week").iloc[-1]
    return {
        'Clear': int(last_weather["Clear"]),
        'Overcast': int(last_weather["Overcast"]),
        'Partially cloudy': int(last_weather["Partially cloudy"]),
        'Rain': int(last_weather["Rain"]),
        'Snow': int(last_weather["Snow"]),
        'tempmax_mean_destino': last_weather["tempmax_mean_destino"],
        'tempmin_mean_destino': last_weather["tempmin_mean_destino"],
        'temp_mean_destino': last_weather["temp_mean_destino"],
        'cloudcover_mean_destino': last_weather["cloudcover_mean_destino"],
        'precip_mean_destino': last_weather["precip_mean_destino"],
    }

def predict_viajeros(provincia_destino_codigo, fecha, is_festivo, is_high_season, weather_conditions):
    model = load_model()
    year = fecha.year
    month = fecha.month
    day_of_week = fecha.weekday()

    input_data = pd.DataFrame({
        'provincia_destino': [provincia_destino_codigo],
        'year': [year],
        'month': [month],
        'day_of_week': [day_of_week],
        'Clear': [weather_conditions.get('Clear', 0)],
        'Overcast': [weather_conditions.get('Overcast', 0)],
        'Partially cloudy': [weather_conditions.get('Partially cloudy', 0)],
        'Rain': [weather_conditions.get('Rain', 0)],
        'Snow': [weather_conditions.get('Snow', 0)],
        'tempmax_mean_destino': [weather_conditions.get('tempmax_mean_destino', 20)],
        'tempmin_mean_destino': [weather_conditions.get('tempmin_mean_destino', 10)],
        'temp_mean_destino': [weather_conditions.get('temp_mean_destino', 15)],
        'cloudcover_mean_destino': [weather_conditions.get('cloudcover_mean_destino', 50)],
        'precip_mean_destino': [weather_conditions.get('precip_mean_destino', 0)],
        'holiday_type_destino': [1 if is_festivo else 0],
        'is_high_season': [1 if is_high_season else 0],
    })

    prediction = model.predict(input_data)[0]
    return prediction

st.title("📊 Predicción de Viajeros 🚗✈️")

province_mapping = load_province_data()
weather_data = load_weather_data()

# Interfaz de usuario
province_name = st.selectbox("📍 Selecciona la provincia destino:", list(province_mapping.keys()))
provincia_destino_codigo = province_mapping[province_name]
fecha = st.date_input("📅 Selecciona una fecha:")
is_festivo = st.radio("🎉 ¿Es un día festivo?", options=["Sí", "No"])
is_high_season = st.radio("🌞 ¿Es temporada alta?", options=["Sí", "No"])

is_festivo_bool = is_festivo == "Sí"
is_high_season_bool = is_high_season == "Sí"

last_weather_conditions = get_last_weather_conditions(provincia_destino_codigo)

with st.expander("⚙️ Añadir más información (condiciones meteorológicas)"):
    st.markdown("### Condiciones meteorológicas detectadas automáticamente:")
    clear = st.checkbox("☀️ Despejado (Clear)", value=bool(last_weather_conditions['Clear']))
    overcast = st.checkbox("☁️ Nublado (Overcast)", value=bool(last_weather_conditions['Overcast']))
    partially_cloudy = st.checkbox("🌤️ Parcialmente nublado (Partially cloudy)", value=bool(last_weather_conditions['Partially cloudy']))
    rain = st.checkbox("🌧️ Lluvia (Rain)", value=bool(last_weather_conditions['Rain']))
    snow = st.checkbox("❄️ Nieve (Snow)", value=bool(last_weather_conditions['Snow']))
    tempmax_mean_destino = st.slider("🌡️ Temperatura máxima (°C):", min_value=-10, max_value=50, value=int(last_weather_conditions['tempmax_mean_destino']))
    tempmin_mean_destino = st.slider("🌡️ Temperatura mínima (°C):", min_value=-10, max_value=50, value=int(last_weather_conditions['tempmin_mean_destino']))
    cloudcover_mean_destino = st.slider("☁️ Cobertura de nubes (%):", min_value=0, max_value=100, value=int(last_weather_conditions['cloudcover_mean_destino']))
    precip_mean_destino = st.slider("💧 Precipitación (mm):", min_value=0, max_value=100, value=int(last_weather_conditions['precip_mean_destino']))

    weather_conditions = {
        'Clear': int(clear),
        'Overcast': int(overcast),
        'Partially cloudy': int(partially_cloudy),
        'Rain': int(rain),
        'Snow': int(snow),
        'tempmax_mean_destino': tempmax_mean_destino,
        'tempmin_mean_destino': tempmin_mean_destino,
        'temp_mean_destino': (tempmax_mean_destino + tempmin_mean_destino) / 2,
        'cloudcover_mean_destino': cloudcover_mean_destino,
        'precip_mean_destino': precip_mean_destino,
    }

# Botón de predicción
if st.button("🔍 Predecir"):
    try:
        prediccion = predict_viajeros(
            provincia_destino_codigo, fecha, is_festivo_bool, is_high_season_bool, weather_conditions
        )
        st.success(f"✅ El número estimado de viajeros es: {int(prediccion)} 🧳")
    except Exception as e:
        st.error(f"❌ Error en la predicción: {e}")

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- UI ---
st.set_page_config(page_title="Prediksi dengan Random Forest", layout="centered")
st.title("⚡ Prediksi Parameter Dryer - Random Forest")
st.image("logo.png", width=150)

gas_input = st.number_input("Masukkan nilai GAS_MMBTU", min_value=0.0, step=0.1, value=16.5)
sm3_input = st.number_input("Masukkan nilai GAS_Sm3", min_value=0.0, step=0.1, value=500.0)

# Daftar model per parameter
parameter_models = {
    'D101330TT (Tem.outlet chamber)': 'saved_models_v2/rf_model_D101330TT (Tem.outlet chamber).pkl',
    'D102260TIC_CV (High press. Steam damper)': 'saved_models_v2/rf_model_D102260TIC_CV (High press. Steam damper).pkl',
    'D102265TIC_CV (Low press. Steam damper)': 'saved_models_v2/rf_model_D102265TIC_CV (Low press. Steam damper).pkl',
    'D102265TIC_PV (Temp. inlet chamber)': 'saved_models_v2/rf_model_D102265TIC_PV (Temp. inlet chamber).pkl',
    'D102266TIC (Main heater dehumidifier)': 'saved_models_v2/rf_model_D102266TIC (Main heater dehumidifier).pkl'
}

if st.button("🔍 Prediksi Sekarang"):
    input_data = np.array([[gas_input, sm3_input]])
    results = []

    for param, model_path in parameter_models.items():
        model = joblib.load(model_path)
        pred = model.predict(input_data)[0]
        results.append({"Parameter": param, "Prediksi": round(pred, 2)})

    result_df = pd.DataFrame(results)
    st.success("📈 Hasil Prediksi:")
    st.table(result_df)

import streamlit as st

st.set_page_config(page_title="Prediksi Dryer", page_icon="🔥", layout="centered")
st.image("logo.png", width=150)

st.title("📊 Prediksi Parameter Dryer")
st.markdown("Silakan pilih model prediksi berdasarkan nilai **GAS_MMBTU & GAS_Sm3**")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌲 Decision Tree"):
        st.switch_page("pages/1_Prediksi_DT.py")

with col2:
    if st.button("🌳 Random Forest"):
        st.switch_page("pages/2_Prediksi_RF.py")

with col3:
    if st.button("⚡ XGBoost"):
        st.switch_page("pages/3_Prediksi_XGB.py")

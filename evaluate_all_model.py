import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("Merged (Gas per jam).csv")
df.dropna(inplace=True)

# Input dan Target
input_features = ['GAS_MMBTU']
target_columns = ['D101330TT (Tem.outlet chamber)','D102265TIC_PV (Temp. inlet chamber)','D102260TIC_CV (High press. Steam damper)','D102265TIC_CV (Low press. Steam damper)','D102266TIC (Main heater dehumidifier)']

# Folder model
model_dir = "saved_models_v2"

# Model type mapping
model_types = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "dt": "Decision Tree"
}

# Menyimpan hasil evaluasi
all_results = []

for model_prefix, model_name in model_types.items():
    for target in target_columns:
        model_path = os.path.join(model_dir, f"{model_prefix}_model_{target}.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)

            # Split X dan y
            X = df[input_features].values
            y_true = df[target].values

            # Prediksi
            y_pred = model.predict(X)

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            mean_actual = np.mean(y_true)
            mean_pred = np.mean(y_pred)
            accuracy_vs_mean = 100 - abs(mean_pred - mean_actual) / mean_actual * 100

            all_results.append({
                "Model": model_name,
                "Target Parameter": target,
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R²": round(r2, 4),
                "Rata-rata Aktual": round(mean_actual, 2),
                "Rata-rata Prediksi": round(mean_pred, 2),
                "Akurasi terhadap Rata-rata (%)": round(accuracy_vs_mean, 2)
            })

# Tampilkan hasil
df_result = pd.DataFrame(all_results)
print(df_result)

# Simpan ke file
df_result.to_csv("evaluasi_semua_model.csv", index=False)
print("\n✅ Hasil evaluasi disimpan ke evaluasi_semua_model.csv")

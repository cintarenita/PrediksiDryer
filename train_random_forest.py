import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import re

# Load data
df = pd.read_csv("Merged (Gas per jam).csv")
df.dropna(inplace=True)

# Fitur input dan target
input_features = ['GAS_MMBTU']
target_columns = [
    'D101330TT (Tem.outlet chamber)',
    'D102265TIC_PV (Temp. inlet chamber)',
    'D102260TIC_CV (High press. Steam damper)',
    'D102265TIC_CV (Low press. Steam damper)',
    'D102266TIC (Main heater dehumidifier)'
]

# Direktori simpan model & metrik
model_dir = "saved_models_rf"
metrics_dir = "saved_metrics_rf"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# Fungsi untuk membersihkan nama file
def sanitize_filename(name):
    name = name.lower().replace(" ", "_")
    name = re.sub(r"[()\.]", "", name)  # hapus tanda kurung dan titik
    name = re.sub(r"[^a-z0-9_]", "", name)  # hanya huruf, angka, dan underscore
    return name

results = []

for target in target_columns:
    X = df[input_features].values  # hanya 1 fitur
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mean_actual = np.mean(y_test)
    mean_pred = np.mean(y_pred)
    accuracy_vs_mean = 100 - abs(mean_pred - mean_actual) / mean_actual * 100

    results.append({
        "Target": target,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Mean_Actual": round(mean_actual, 2),
        "Mean_Predicted": round(mean_pred, 2),
        "Accuracy_vs_Mean (%)": round(accuracy_vs_mean, 2)
    })

    # Simpan model
    safe_target_name = sanitize_filename(target)
    model_filename = f"{model_dir}/rf_model_{safe_target_name}.pkl"

    # Hapus model lama (overwrite)
    if os.path.exists(model_filename):
        os.remove(model_filename)

    joblib.dump(model, model_filename)
    print(f"✅ Model {target} disimpan dengan {model.n_features_in_} fitur ➜ {model_filename}")

# Simpan hasil evaluasi
results_df = pd.DataFrame(results)
results_df.to_csv(f"{metrics_dir}/random_forest_metrics.csv", index=False)
print("✅ Semua model Random Forest dan evaluasi berhasil disimpan.")

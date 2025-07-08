import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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
model_dir = "saved_models_v2"
metrics_dir = "saved_metrics_v2"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# Fungsi membersihkan nama file
def sanitize_filename(name):
    name = name.lower().replace(" ", "_")
    name = re.sub(r"[()\.]", "", name)  # hapus tanda kurung dan titik
    name = re.sub(r"[^a-z0-9_]", "", name)  # hanya huruf, angka, underscore
    return name

results = []

for target in target_columns:
    X = df[input_features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=42)
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

    # Simpan model dengan nama aman
    safe_target_name = sanitize_filename(target)
    model_filename = f"{model_dir}/dt_model_{safe_target_name}.pkl"
    joblib.dump(model, model_filename)

# Simpan evaluasi ke CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f"{metrics_dir}/decision_tree_metrics.csv", index=False)
print("✅ Decision Tree model dan evaluasi disimpan.")

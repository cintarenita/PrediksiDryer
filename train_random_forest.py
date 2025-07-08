# 1. Import library
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 2. Load dataset
df = pd.read_csv("Merged (Gas per jam).csv")

# 3. Pilih fitur dan target
selected_columns = [
    'GAS_MMBTU',
    'D101330TT (Tem.outlet chamber)',
    'D102265TIC_PV (Temp. inlet chamber)',
    'D102260TIC_CV (High press. Steam damper)',
    'D102265TIC_CV (Low press. Steam damper)',
    'D102266TIC (Main heater dehumidifier)'
]
df_clean = df[selected_columns].dropna()

X = df_clean[['GAS_MMBTU']]
y = df_clean.drop(columns=['GAS_MMBTU'])

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model training
rf_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
rf_model.fit(X_train, y_train)

# 6. Save model
joblib.dump(rf_model, "model_RF_v1.7.0.pkl")

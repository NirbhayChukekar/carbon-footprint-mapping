import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV exported from GEE
df = pd.read_csv('carbon_features.csv').dropna()
print(f"Rows loaded: {len(df)}")

X = df[['ndvi', 'lulc', 'temperature', 'population']]
y = df['carbon']

# Preprocessing
pre = ColumnTransformer([
    ('num', StandardScaler(),                       ['ndvi', 'temperature', 'population']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['lulc']),
])

# Model
model = Pipeline([
    ('pre', pre),
    ('rf',  RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, pred)):.2f}")
print(f"R²   : {r2_score(y_test, pred):.4f}")

# Save
Path('models').mkdir(exist_ok=True)
with open('models/carbon_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Saved → models/carbon_model.pkl")
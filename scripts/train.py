import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Get the project root directory (parent of scripts folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 1. Load Data
data_path = os.path.join(project_root, 'data', 'Sleep_health_and_lifestyle_dataset.csv')
df = pd.read_csv(data_path)
print(f"Loaded data from: {data_path}")

# 2. Preprocessing
df['Sleep_Category'] = df['Quality of Sleep'].apply(lambda x: 1 if x >= 7 else 0)
features = ['Age', 'Sleep Duration', 'Physical Activity Level', 'Heart Rate', 'Daily Steps']
X = df[features]
y = df['Sleep_Category']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model with specific constraints to prevent overfitting
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. FEATURE IMPORTANCE (Show this to Mam!)
importances = pd.Series(model.feature_importances_, index=features)
print("\n--- Feature Importance ---")
print(importances.sort_values(ascending=False))

# 6. Save Model to models folder
model_path = os.path.join(project_root, 'models', 'sleep_quality_model.pkl')
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")
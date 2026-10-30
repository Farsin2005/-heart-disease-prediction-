import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Dataset (Standard UCI Heart Disease Dataset format)
# Ensure 'heart.csv' is in your directory
df = pd.read_csv('heart.csv')

# 2. Split Features and Target
# 'target' or 'output' column: 1 = Disease risk, 0 = Healthy
X = df.drop(columns=['target'])
y = df['target']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Accuracy
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. Save the trained model and column mapping
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')
print("Model saved successfully as 'heart_disease_model.pkl'!")

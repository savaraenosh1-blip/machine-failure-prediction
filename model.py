import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import json

# Load dataset
df = pd.read_csv("machine_failure_dataset.csv")

# Drop unnecessary columns
df.drop(columns=["UDI", "Product ID"], errors='ignore', inplace=True)

# Convert categorical to numeric
df = pd.get_dummies(df, columns=["Type"])

# ❌ REMOVE FAILURE-RELATED FEATURES (VERY IMPORTANT)
df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"], inplace=True)

# Features & target
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save columns
columns = list(X.columns)
with open("columns.json", "w") as f:
    json.dump(columns, f)

print("✅ Model trained correctly and saved!")
print("Columns:", columns)
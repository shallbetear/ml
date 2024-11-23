import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)

num_samples = 400
columns = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", 
    "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", 
    "dm", "cad", "appet", "pe", "ane", "classification"
]

data = {
    "age": np.random.randint(20, 80, size=num_samples),
    "bp": np.random.randint(50, 180, size=num_samples),
    "sg": np.round(np.random.uniform(1.005, 1.025, size=num_samples), 3),
    "al": np.random.randint(0, 5, size=num_samples),
    "su": np.random.randint(0, 5, size=num_samples),
    "rbc": np.random.choice([1, 0], size=num_samples, p=[0.3, 0.7]),
    "pc": np.random.choice([1, 0], size=num_samples, p=[0.6, 0.4]),
    "pcc": np.random.choice([1, 0], size=num_samples, p=[0.1, 0.9]),
    "ba": np.random.choice([1, 0], size=num_samples, p=[0.1, 0.9]),
    "bgr": np.random.randint(70, 400, size=num_samples),
    "bu": np.random.randint(10, 150, size=num_samples),
    "sc": np.round(np.random.uniform(0.4, 15.0, size=num_samples), 2),
    "sod": np.round(np.random.uniform(120, 150, size=num_samples), 1),
    "pot": np.round(np.random.uniform(3.5, 6.0, size=num_samples), 1),
    "hemo": np.round(np.random.uniform(7.0, 17.0, size=num_samples), 1),
    "pcv": np.random.randint(20, 50, size=num_samples),
    "wbcc": np.random.randint(4000, 12000, size=num_samples),
    "rbcc": np.round(np.random.uniform(3.5, 6.5, size=num_samples), 1),
    "htn": np.random.choice([1, 0], size=num_samples, p=[0.5, 0.5]),
    "dm": np.random.choice([1, 0], size=num_samples, p=[0.4, 0.6]),
    "cad": np.random.choice([1, 0], size=num_samples, p=[0.1, 0.9]),
    "appet": np.random.choice([1, 0], size=num_samples, p=[0.8, 0.2]),
    "pe": np.random.choice([1, 0], size=num_samples, p=[0.1, 0.9]),
    "ane": np.random.choice([1, 0], size=num_samples, p=[0.2, 0.8]),
    "classification": np.random.choice([1, 0], size=num_samples, p=[0.3, 0.7])
}

df = pd.DataFrame(data)

print(df.head())

df.replace('?', np.nan, inplace=True) 
df.fillna(df.mode().iloc[0], inplace=True) 

X = df.drop('classification', axis=1) 
y = df['classification']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

new_data = np.array([[30, 80, 1.015, 1, 0, 1, 1, 0, 0, 150, 40, 1.1, 135, 4.5, 12.0, 30, 6000, 4.5, 1, 0, 0, 1, 0, 0]]) 
new_data_scaled = scaler.transform(new_data) 
prediction = model.predict(new_data_scaled)

print("Predicted Kidney Disease: ", "Positive" if prediction[0] == 1 else "Negative")

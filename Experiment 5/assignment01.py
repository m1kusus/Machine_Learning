import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = {
    "Study_Hours": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
                    2, 3, 4, 5, 6, 3, 7, 8, 5, 4],
    "Attendance": [40, 55, 45, 60, 50, 65, 60, 70, 65, 75, 70, 80, 75, 82, 80, 88, 85, 90, 90, 95,
                   48, 52, 58, 72, 68, 80, 63, 77, 85, 55],
    "Pass": [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nActual Values:")
print(y_test.values)
print("\nPredicted Values:")
print(y_pred)

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")

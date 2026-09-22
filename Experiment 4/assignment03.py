import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("Housing.csv")

X = df[["area"]]
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

linear_r2 = r2_score(y_test, y_pred)
print(f"Linear Regression R2 : {linear_r2:.4f}")

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

poly_r2 = r2_score(y_test, y_pred_poly)
print(f"Polynomial Regression R2 : {poly_r2:.4f}")


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

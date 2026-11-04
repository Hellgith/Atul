import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("uber.csv")

df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")
df = df.dropna()
df = df[df["fare_amount"].between(1, 100)]

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)
df["hour"] = df["pickup_datetime"].dt.hour
df = df.drop("pickup_datetime", axis=1)

lat1 = np.radians(df["pickup_latitude"])
lat2 = np.radians(df["dropoff_latitude"])
lon1 = np.radians(df["pickup_longitude"])
lon2 = np.radians(df["dropoff_longitude"])

a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
df["distance"] = 6371 * 2 * np.arcsin(np.sqrt(a))

df = df[df["distance"] > 0].copy()

q1 = df["fare_amount"].quantile(0.25)
q3 = df["fare_amount"].quantile(0.75)
iqr = q3 - q1

print("Outliers:", ((df["fare_amount"] < q1-1.5*iqr) |
                    (df["fare_amount"] > q3+1.5*iqr)).sum())

df = df[df["fare_amount"].between(q1-1.5*iqr, q3+1.5*iqr)]

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

X = df.drop("fare_amount", axis=1)
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)

    print("\n", name)
    print("R2 Score:", round(r2_score(y_test, p), 3))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, p)), 3))
    print("MAE:", round(mean_absolute_error(y_test, p), 3))
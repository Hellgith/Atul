import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(
    "/content/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print(data.index)
print(data)
print(data.columns)
print(data.shape)
print(data.head())
print(data.tail())
print(data.nunique())

print(data.isna().sum())
print(data.isnull().sum())

print(
    "Number of rows before removing duplicates:",
    len(data)
)

data_cleaned = data.drop_duplicates()

print(
    "Number of rows after removing duplicates:",
    len(data_cleaned)
)

print(data.describe())

# Unique values and counts

unique, counts = np.unique(
    data["tenure"],
    return_counts=True
)

print(unique)
print(counts)

unique, counts = np.unique(
    data["MonthlyCharges"],
    return_counts=True
)

print(unique)
print(counts)

unique, counts = np.unique(
    data["TotalCharges"],
    return_counts=True
)

print(unique)
print(counts)

# Pairplot

sns.pairplot(data)
plt.show()

# Boxplot - Tenure

plt.boxplot(data["tenure"])
plt.show()

# Boxplot - Monthly Charges

plt.boxplot(data["MonthlyCharges"])
plt.show()

# Split data

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Save cleaned data

data.to_csv(
    "Cleaned_Telecom_Customer_Churn.csv",
    index=False
)
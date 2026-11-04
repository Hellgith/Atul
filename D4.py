import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load dataset

df = pd.read_csv("RealEstate_Prices.csv")

# Basic information

print(df.head())
print(df.shape)
print(df.columns)

# Clean column names

df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("(", "", regex=False)
df.columns = df.columns.str.replace(")", "", regex=False)

# Missing values

print(df.isnull().sum())

# Handle missing values

df["Bedrooms"] = df["Bedrooms"].fillna(
    df["Bedrooms"].mean()
)

df["Sale_Price"] = df["Sale_Price"].fillna(
    df["Sale_Price"].mean()
)

df["Location"] = df["Location"].fillna(
    df["Location"].mode()[0]
)

print(df.isnull().sum())

# Merge additional dataset

neighborhood = pd.read_csv("Neighborhood_Data.csv")

df = pd.merge(
    df,
    neighborhood,
    on="Location",
    how="left"
)

# Filtering

pune_properties = df[
    df["Location"] == "Pune"
]

expensive_houses = df[
    df["Sale_Price"] > 5000000
]

apartments = df[
    df["Property_Type"] == "Apartment"
]

# Categorical encoding

encoder = LabelEncoder()

df["Property_Type"] = encoder.fit_transform(
    df["Property_Type"]
)

# Aggregation

print(
    df.groupby("Location")["Sale_Price"].mean()
)

print(
    df.groupby("Property_Type")["Sale_Price"].mean()
)

print(
    df.groupby("Location").size()
)

# Derived metric

df["Price_Per_SqFt"] = (
    df["Sale_Price"] / df["Area_sq_ft"]
)

# Outlier detection

sns.boxplot(x=df["Sale_Price"])
plt.show()

Q1 = df["Sale_Price"].quantile(0.25)
Q3 = df["Sale_Price"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Remove outliers

df = df[
    (df["Sale_Price"] >= lower_limit) &
    (df["Sale_Price"] <= upper_limit)
]

# Visualization

df.groupby("Location")["Sale_Price"].mean().plot(
    kind="bar"
)

plt.title("Average Property Prices by Location")
plt.ylabel("Price")
plt.show()

# Save cleaned data

df.to_csv(
    "Cleaned_RealEstate_Prices.csv",
    index=False
)
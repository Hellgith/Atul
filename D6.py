
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Retail_Sales_Data.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

print(df[["Region", "ProductCategory", "SalesAmount"]].head())
print("\nMissing Values:\n", df.isnull().sum())

df = df.dropna().drop_duplicates()

region_sales = df.groupby("Region")["SalesAmount"].sum().reset_index()
print("\nSales by Region:\n", region_sales)

plt.bar(region_sales["Region"], region_sales["SalesAmount"])
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales Amount")
plt.show()

plt.pie(region_sales["SalesAmount"], labels=region_sales["Region"], autopct="%1.1f%%")
plt.title("Sales Distribution by Region")
plt.show()

top_regions = region_sales.sort_values("SalesAmount", ascending=False)
print("\nTop Performing Region:\n", top_regions.iloc[0])

rc_sales = df.groupby(["Region", "ProductCategory"])["SalesAmount"].sum().reset_index()
print("\nRegion and Category Sales:\n", rc_sales)

pivot = rc_sales.pivot(index="Region", columns="ProductCategory", values="SalesAmount")

pivot.plot(kind="bar", stacked=True, figsize=(10,5))
plt.title("Sales by Region and Product Category")
plt.xlabel("Region")
plt.ylabel("Sales Amount")
plt.show()

pivot.plot(kind="bar", figsize=(10,5))
plt.title("Regional Sales Comparison")
plt.xlabel("Region")
plt.ylabel("Sales Amount")
plt.show()

print("\nAverage Sales:\n", df.groupby("Region")["SalesAmount"].mean())
print("\nTotal Quantity Sold:\n", df.groupby("Region")["QuantitySold"].sum())

region_sales.to_csv("Regional_Sales_Summary.csv", index=False)

print("\nFile Saved Successfully!")

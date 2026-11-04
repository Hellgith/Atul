# import json

# sales_data = [
#     {
#         "order_id": 101,
#         "product": "Laptop",
#         "quantity": 2,
#         "Price": 50000
#     },
#     {
#         "order_id": 102,
#         "product": "Mobile",
#         "quantity": 3,
#         "Price": 15000
#     }
# ]

# # Calculate total sales
# for sale in sales_data:
#     sale["total_sales"] = sale["quantity"] * sale["price"]

# # Save as JSON file
# with open("sales_dataset.json", "w") as file:
#     json.dump(sales_data, file, indent=4)

import pandas as pd

# -----------------------------
# Load Data
# -----------------------------

csv_data = pd.read_csv("sales.csv")
excel_data = pd.read_excel("sales.xlsx")
json_data = pd.read_json("sales_dataset.json")

# -----------------------------
# Combine Data
# -----------------------------

sales = pd.concat(
    [csv_data, excel_data, json_data],
    ignore_index=True
)

# -----------------------------
# Data Cleaning
# -----------------------------

sales.drop_duplicates(inplace=True)
sales.dropna(inplace=True)

# -----------------------------
# Create TotalSales Column
# -----------------------------

sales["TotalSales"] = sales["Quantity"] * sales["Price"]

# -----------------------------
# Display Dataset
# -----------------------------

print("\nCombined Sales Data")
print(sales)

# -----------------------------
# Dataset Information
# -----------------------------

print("\nDataset Information")
print(sales.info())

# -----------------------------
# Summary Statistics
# -----------------------------

print("\nSummary Statistics")
print(sales.describe())

# -----------------------------
# Total Revenue
# -----------------------------

total_revenue = sales["TotalSales"].sum()

print("\nTotal Revenue =", total_revenue)

# -----------------------------
# Sales by Category
# -----------------------------

category_sales = sales.groupby("Category")["TotalSales"].sum()

print("\nSales by Category")
print(category_sales)

# -----------------------------
# Product-wise Sales
# -----------------------------

product_sales = sales.groupby("Product")["TotalSales"].sum()

print("\nProduct-wise Sales")
print(product_sales)

# -----------------------------
# Top Selling Products
# -----------------------------

top_products = (
    sales.groupby("Product")["TotalSales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop Selling Products")
print(top_products)

# -----------------------------
# Average Price by Category
# -----------------------------

avg_price = sales.groupby("Category")["Price"].mean()

print("\nAverage Price by Category")
print(avg_price)

# -----------------------------
# Export Cleaned Data
# -----------------------------

sales.to_csv("combined_sales.csv", index=False)

print("\ncombined_sales.csv has been created successfully.")
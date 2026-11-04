
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("City_Air_Quality.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

df = df.dropna().drop_duplicates()
df["Date"] = pd.to_datetime(df["Date"])

plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["AQI"], color="red", marker="o", label="AQI")
plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid()
plt.show()

for col in ["PM2.5", "PM10", "CO"]:
    plt.figure(figsize=(8,5))
    plt.plot(df["Date"], df[col], label=col)
    plt.title(col + " Trend")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.legend()
    plt.show()

plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["PM2.5"], label="PM2.5")
plt.plot(df["Date"], df["PM10"], label="PM10")
plt.plot(df["Date"], df["CO"], label="CO")
plt.title("Pollutant Levels")
plt.xlabel("Date")
plt.ylabel("Pollutant Level")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.bar(df["Date"].head(10), df["AQI"].head(10))
plt.title("AQI Comparison")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.show()

d = df.head(10)
plt.bar(d["Date"], d["PM2.5"], label="PM2.5")
plt.bar(d["Date"], d["PM10"], bottom=d["PM2.5"], label="PM10")
plt.title("Stacked Pollutant Levels")
plt.legend()
plt.show()

plt.boxplot(df["AQI"])
plt.title("AQI Distribution - Box Plot")
plt.show()

sns.violinplot(y=df["AQI"])
plt.title("AQI Distribution - Violin Plot")
plt.show()

plt.scatter(df["PM2.5"], df["AQI"])
plt.title("AQI vs PM2.5")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.show()

plt.scatter(df["PM2.5"], df["AQI"], s=df["CO"]*100)
plt.title("AQI Bubble Chart")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.show()

sns.heatmap(df[["PM2.5","PM10","CO","AQI"]].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Between Pollutants and AQI")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df["Date"], df["AQI"], color="red")
plt.title("AQI Trend")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.savefig("AQI_Trend.png")
plt.show()

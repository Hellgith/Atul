import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

api_key = "3a0715e329e888619c803200728b1fb8"
city = "Pune"

url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

response = requests.get(url)
data = response.json()

print("API Response:")
print(data)

temperature = data["main"]["temp"]
humidity = data["main"]["humidity"]
wind_speed = data["wind"]["speed"]
weather = data["weather"][0]["description"]

print("Temperature:", temperature)
print("Humidity:", humidity)
print("Wind Speed:", wind_speed)
print("Weather:", weather)

api_weather_df = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "WindSpeed": [wind_speed],
    "Weather": [weather]
})

print("\nAPI Weather Data:")
print(api_weather_df)

weather_data = {
    "Date": [
        "2024-06-01",
        "2024-06-02",
        "2024-06-03",
        "2024-06-04",
        "2024-06-05"
    ],
    "Temperature": [31, 33, 29, 32, 30],
    "Humidity": [60, 65, 70, 58, 62],
    "WindSpeed": [4, 5, 6, 3, 4]
}

weather_df = pd.DataFrame(weather_data)

print("\nSample Weather Dataset:")
print(weather_df)

print("\nMissing Values:")
print(weather_df.isnull().sum())

weather_df["Date"] = pd.to_datetime(weather_df["Date"])

print("\nData after Date Conversion:")
print(weather_df)

avg_temp = weather_df["Temperature"].mean()

print("\nAverage Temperature =", avg_temp)
print("Maximum Temperature =", weather_df["Temperature"].max())
print("Minimum Temperature =", weather_df["Temperature"].min())

# Temperature Trend

plt.figure(figsize=(8, 5))
plt.plot(
    weather_df["Date"],
    weather_df["Temperature"],
    marker="o"
)
plt.title("Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Humidity Levels

plt.figure(figsize=(8, 5))
plt.bar(
    weather_df["Date"],
    weather_df["Humidity"]
)
plt.title("Humidity Levels")
plt.xlabel("Date")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Temperature vs Humidity

plt.figure(figsize=(7, 5))
plt.scatter(
    weather_df["Temperature"],
    weather_df["Humidity"]
)
plt.title("Temperature vs Humidity")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.tight_layout()
plt.show()

# Daily Average Temperature

daily_avg = weather_df.groupby("Date")["Temperature"].mean()

print("\nDaily Average Temperature:")
print(daily_avg)

# Monthly Average Temperature

weather_df["Month"] = weather_df["Date"].dt.month_name()

monthly_avg = weather_df.groupby("Month")["Temperature"].mean()

print("\nMonthly Average Temperature:")
print(monthly_avg)

# Correlation Matrix

print("\nCorrelation Matrix:")
print(weather_df.corr(numeric_only=True))

# Weather Correlation Heatmap

plt.figure(figsize=(7, 5))

sns.heatmap(
    weather_df.corr(numeric_only=True),
    annot=True
)

plt.title("Weather Correlation Heatmap")
plt.tight_layout()
plt.show()
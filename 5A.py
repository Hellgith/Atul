import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("car_evaluation.csv", header=None)

df.columns = ["buying", "maint", "doors", "persons",
              "lug_boot", "safety", "class"]

encoders = {}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=encoders["class"].classes_
))

car = pd.DataFrame(
    [["low", "low", "4", "more", "big", "high"]],
    columns=X.columns
)

for col in car.columns:
    car[col] = encoders[col].transform(car[col])

prediction = model.predict(car)

print("\nPredicted Car Safety Class:",
      encoders["class"].inverse_transform(prediction)[0])

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=encoders["class"].classes_,
    yticklabels=encoders["class"].classes_
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()
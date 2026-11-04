import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("Iris.csv")
df = df.drop("Id", axis=1)

X = df.drop("Species", axis=1)
y = df["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base = [
    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("dt", DecisionTreeClassifier(random_state=42)),
    ("svm", make_pipeline(StandardScaler(), SVC(probability=True)))
]

models = {
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="mlogloss", random_state=42
    ),
    "Hard Voting": VotingClassifier(base, voting="hard"),
    "Soft Voting": VotingClassifier(base, voting="soft")
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, average="weighted"),
        recall_score(y_test, pred, average="weighted"),
        f1_score(y_test, pred, average="weighted")
    ])

    cm = confusion_matrix(y_test, pred)

    print("\n", name)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Setosa", "Versicolor", "Virginica"],
                yticklabels=["Setosa", "Versicolor", "Virginica"])
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.round(3).to_string(index=False))

results_df.set_index("Model").plot(
    kind="bar", figsize=(12, 6), ylim=(0, 1.1)
)

plt.title("Performance Comparison of Ensemble Models")
plt.xlabel("Models")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
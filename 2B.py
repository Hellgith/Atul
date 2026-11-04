import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score

uci = load_diabetes(as_frame=True).frame
uci.rename(columns={"target": "Outcome"}, inplace=True)

pima = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

def analyze(df, name, binary=False):
    print("\n", name)

    stats = []
    for col in df.select_dtypes(include=np.number):
        x = df[col].dropna()
        stats.append({
            "Variable": col,
            "Frequency": x.value_counts().iloc[0],
            "Mean": round(x.mean(), 2),
            "Median": round(x.median(), 2),
            "Mode": round(x.mode().iloc[0], 2),
            "Variance": round(x.var(), 2),
            "Std Dev": round(x.std(), 2),
            "Skewness": round(skew(x), 2),
            "Kurtosis": round(kurtosis(x), 2)
        })

    print("\nUnivariate Analysis:")
    print(pd.DataFrame(stats).to_string(index=False))

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

    print("\nLinear Regression R2:", round(r2_score(y_test, pred), 3))

    mr = LinearRegression()
    mr.fit(X_train, y_train)
    print("Multiple Regression R2:", round(r2_score(y_test, mr.predict(X_test)), 3))

    if binary:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        log = LogisticRegression(max_iter=1000)
        log.fit(X_train, y_train)

        print("Logistic Regression Accuracy:",
              round(accuracy_score(y_test, log.predict(X_test)) * 100, 2), "%")

analyze(uci, "UCI DIABETES DATASET")
analyze(pima, "PIMA INDIANS DIABETES DATASET", True)
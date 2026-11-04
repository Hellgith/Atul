import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score

df = pd.read_csv("CC GENERAL.csv")
data = df.sample(n=800, random_state=42).reset_index(drop=True)
X = data.drop(columns=["CUST_ID"], errors="ignore")
X = SimpleImputer(strategy="median").fit_transform(X)
X = StandardScaler().fit_transform(X)
D = pairwise_distances(X)

def kmedoids(D, k, max_iter=30):
    n = len(D)
    medoids = [np.argmin(D.sum(axis=1))]
    while len(medoids) < k:
        dist = D[:, medoids].min(axis=1)
        dist[medoids] = -1
        medoids.append(np.argmax(dist))
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = []
        for i in range(k):
            members = np.where(labels == i)[0]
            if len(members) == 0:
                new_medoids.append(medoids[i])
            else:
                costs = D[np.ix_(members, members)].sum(axis=1)
                new_medoids.append(members[np.argmin(costs)])
        if new_medoids == medoids:
            break
        medoids = new_medoids
    return np.argmin(D[:, medoids], axis=1), medoids

scores = []
ks = range(2, 7)

for k in ks:
    labels, medoids = kmedoids(D, k)
    score = silhouette_score(D, labels, metric="precomputed")
    scores.append(score)
    print("K =", k, "Silhouette Score =", round(score, 3))

plt.plot(list(ks), scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

best_k = list(ks)[np.argmax(scores)]
labels, medoids = kmedoids(D, best_k)
data["Cluster"] = labels

print("Optimal K:", best_k)
print("Cluster Sizes:\n", data["Cluster"].value_counts().sort_index())
print("Medoid Customer IDs:", data.iloc[medoids]["CUST_ID"].tolist())

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv")

X = df.drop("Customer_Segment", axis=1)
y = df["Customer_Segment"]

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained Variance:", pca.explained_variance_ratio_)
print("Total Variance (%):", sum(pca.explained_variance_ratio_) * 100)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on Wine Dataset")
plt.show()

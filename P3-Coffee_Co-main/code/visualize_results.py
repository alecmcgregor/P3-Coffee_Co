import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load CSV files
lin = pd.read_csv("linear_predictions.csv")
tree = pd.read_csv("tree_predictions.csv")
lin_weights = pd.read_csv("linear_weights.csv", header=None, names=["Weight"])
tree_imports = pd.read_csv("tree_importances.csv", header=None, names=["Importance"])

labels = [
    "Aroma", "Aftertaste", "Acidity", "Body",
    "Balance", "Uniformity", "Sweetness", "Moisture"
]

# ---------- Linear Regression Visuals ----------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(lin["Actual"], lin["Predicted"], alpha=0.6, color='dodgerblue')
plt.plot([lin["Actual"].min(), lin["Actual"].max()],
         [lin["Actual"].min(), lin["Actual"].max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression")

plt.subplot(1, 3, 2)
plt.bar(labels, lin_weights["Weight"])
plt.xticks(rotation=45)
plt.title("Linear Weights")

plt.subplot(1, 3, 3)
residuals = lin["Actual"] - lin["Predicted"]
sns.histplot(residuals, kde=True)
plt.title("Linear Residuals")

plt.tight_layout()
plt.show()

# ---------- Decision Tree Visuals ----------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(tree["Actual"], tree["Predicted"], alpha=0.6, color='forestgreen')
plt.plot([tree["Actual"].min(), tree["Actual"].max()],
         [tree["Actual"].min(), tree["Actual"].max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Decision Tree")

plt.subplot(1, 3, 2)
plt.bar(labels, tree_imports["Importance"])
plt.xticks(rotation=45)
plt.title("Tree Feature Importances")

plt.subplot(1, 3, 3)
residuals = tree["Actual"] - tree["Predicted"]
sns.histplot(residuals, kde=True)
plt.title("Tree Residuals")

plt.tight_layout()
plt.show()

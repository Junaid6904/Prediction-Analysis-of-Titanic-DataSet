# Titanic Survival Prediction Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset
data = pd.read_csv("C:/VOLUME_D/EXCEL/train.csv")

# Step 2: Show first 5 rows
print(data.head())

# Step 3: Check missing values
print(data.isnull().sum())

# Step 4: Fill missing values
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

# Step 5: Create FamilySize column
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

# -----------------------------
# Graphs
# -----------------------------

# Survival count graph
plt.figure()
sns.countplot(x="Survived", data=data)
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# Gender graph with clear labels
plt.figure()
sns.countplot(x="Sex", hue="Survived", data=data)
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Age histogram
plt.figure()
plt.hist(data["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Fare histogram
plt.figure()
plt.hist(data["Fare"], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Model part
# -----------------------------

# Convert Sex into numbers for machine learning
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# Step 6: Select useful columns
x = data[["Pclass", "Sex", "Age", "Fare", "FamilySize"]]
y = data["Survived"]

# Step 7: Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Step 8: Create model
model = LogisticRegression(max_iter=1000)

# Step 9: Train model
model.fit(x_train, y_train)

# Step 10: Predict
y_pred = model.predict(x_test)

# Step 11: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy is:", accuracy)

# Step 12: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Heatmap
plt.figure()
sns.heatmap(data[["Survived", "Pclass", "Sex", "Age", "Fare", "FamilySize"]].corr(), annot=True)
plt.title("Heatmap")
plt.show()

# Confusion Matrix graph
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save image
plt.savefig("ConfusionMatrix.png")

plt.show()
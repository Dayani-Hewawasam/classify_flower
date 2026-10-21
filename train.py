# --- 1. IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# --- 2. LOAD DATA ---
iris = load_iris(as_frame=True)
df = iris.frame                      # a pandas DataFrame
print(df.head())                     # look at the first 5 rows
print(df["target"].value_counts())   # how many samples per class (0,1,2)

# --- 3. SEPARATE FEATURES (X) AND LABEL (y) ---
X = df.drop(columns="target")        # inputs: sepal/petal length & width
y = df["target"]                     # output: species 0, 1 or 2

# --- 4. SPLIT INTO TRAIN AND TEST ---
# The model learns from 80% and is judged on the 20% it has never seen.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# random_state=42 -> same split every run (reproducible)
# stratify=y      -> each split keeps the same class proportions

# --- 5. CREATE AND TRAIN THE MODEL ---
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)          # "learning" happens here

# --- 6. EVALUATE ON UNSEEN DATA ---
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("Confusion matrix:\n", confusion_matrix(y_test, preds))

# --- 7. VISUALIZE ---
plt.scatter(df["petal length (cm)"], df["petal width (cm)"], c=df["target"])
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("Iris classes")
plt.savefig("plot.png")

# --- 8. SAVE THE TRAINED MODEL ---
joblib.dump(model, "model.pkl")
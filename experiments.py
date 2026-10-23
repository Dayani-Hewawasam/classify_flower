# experiments.py
# Runs many small experiments on the Iris data, changes ONE thing at a time,
# and stores every accuracy in experiments.csv + a chart in experiments.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- LOAD DATA ---------------------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame
X_all = df.drop(columns="target")   # the 4 measurements
y = df["target"]                    # the species (0, 1, 2)

rows = []   # every result gets added here, then saved to a CSV at the end


# --- ONE FUNCTION THAT TRAINS + TESTS + RECORDS ----------------------------
def run(experiment, setting, X, model,
        test_size=0.2, random_state=42,
        train_fraction=1.0, shuffle_labels=False):
    """Split the data, train the model, measure accuracy, store the result."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # use only part of the training data (to see the effect of less data)
    if train_fraction < 1.0:
        X_train = X_train.sample(frac=train_fraction, random_state=0)
        y_train = y_train.loc[X_train.index]

    # destroy the link between features and labels (sanity check)
    if shuffle_labels:
        y_train = y_train.sample(frac=1, random_state=0).values

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    rows.append({"experiment": experiment, "setting": setting,
                 "accuracy": round(acc, 4)})
    print(f"{experiment:16s} {str(setting):14s} accuracy = {acc:.4f}")


# --- EXPERIMENT 1: number of neighbors (k) ---------------------------------
for k in [1, 3, 5, 9, 15, 30, 50]:
    run("k", k, X_all, KNeighborsClassifier(n_neighbors=k))

# --- EXPERIMENT 2: size of the test set ------------------------------------
for ts in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    run("test_size", ts, X_all, KNeighborsClassifier(n_neighbors=5),
        test_size=ts)

# --- EXPERIMENT 3: how much training data the model gets -------------------
for frac in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
    run("train_fraction", frac, X_all, KNeighborsClassifier(n_neighbors=5),
        train_fraction=frac)

# --- EXPERIMENT 4: adding noise to the measurements ------------------------
for sigma in [0, 0.1, 0.3, 0.5, 1.0, 2.0]:
    rng = np.random.RandomState(0)
    X_noisy = X_all + rng.normal(0, sigma, X_all.shape)
    run("noise", sigma, X_noisy, KNeighborsClassifier(n_neighbors=5))

# --- EXPERIMENT 5: different random splits (shows the spread) --------------
for rs in range(10):
    run("random_state", rs, X_all, KNeighborsClassifier(n_neighbors=5),
        random_state=rs)

# --- EXPERIMENT 6: which features are used ---------------------------------
feature_sets = {
    "all 4": list(X_all.columns),
    "petal only": ["petal length (cm)", "petal width (cm)"],
    "sepal only": ["sepal length (cm)", "sepal width (cm)"],
}
for name, cols in feature_sets.items():
    run("features", name, X_all[cols], KNeighborsClassifier(n_neighbors=5))

# --- EXPERIMENT 7: different models ----------------------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision tree": DecisionTreeClassifier(random_state=42),
    "Logistic reg.": LogisticRegression(max_iter=200),
}
for name, m in models.items():
    run("model", name, X_all, m)

# --- EXPERIMENT 8: sanity check with shuffled labels -----------------------
run("shuffled_labels", "real labels", X_all, KNeighborsClassifier(n_neighbors=5))
run("shuffled_labels", "shuffled", X_all, KNeighborsClassifier(n_neighbors=5),
    shuffle_labels=True)

# --- SAVE EVERYTHING -------------------------------------------------------
results = pd.DataFrame(rows)
results.to_csv("experiments.csv", index=False)

print("\nSummary (min / max / mean accuracy per experiment):")
print(results.groupby("experiment", sort=False)["accuracy"]
      .agg(["min", "max", "mean"]).round(3))

# --- DRAW ONE CHART PER EXPERIMENT -----------------------------------------
panels = [("k", "line"), ("test_size", "line"), ("train_fraction", "line"),
          ("noise", "line"), ("random_state", "line"), ("features", "bar"),
          ("model", "bar"), ("shuffled_labels", "bar")]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, (name, kind) in zip(axes.flat, panels):
    d = results[results["experiment"] == name]
    labels = d["setting"].astype(str)
    if kind == "line":
        ax.plot(labels, d["accuracy"], marker="o")
    else:
        ax.bar(labels, d["accuracy"])
    ax.set_title(name)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("experiments.png")
print("\nSaved: experiments.csv and experiments.png")

import joblib
import pandas as pd

model = joblib.load("model.pkl")
names = ["setosa", "versicolor", "virginica"]

# one new flower: sepal length, sepal width, petal length, petal width
cols = ["sepal length (cm)", "sepal width (cm)",
        "petal length (cm)", "petal width (cm)"]
sample = pd.DataFrame([[5.0, 3.5, 1.3, 0.2]], columns=cols)

print("Predicted species:", names[model.predict(sample)[0]])
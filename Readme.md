# Iris Flower Classifier

## 1. What is this project?

A small machine learning project that predicts the **species of an iris flower** (setosa, versicolor or virginica) from 4 measurements:

- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)

This is called **classification**: the model chooses a category. It uses **supervised learning**, which means the model learns from examples that already have the correct answer.

The goal is not a clever model. The goal is to **understand every step** of a complete ML project, and to see how accuracy changes when the data and settings change.

**Tools:** Python, scikit-learn, pandas, matplotlib, joblib, VS Code, Git.

**Data:** the Iris dataset (150 flowers, 3 species, 50 each). It is bundled inside scikit-learn, so there is nothing to download.

## 2. What I did

### Step 1: Set up the project

```bash
mkdir iris-ml && cd iris-ml
git init
python -m venv .venv
# Windows:    .venv\Scripts\activate
# Mac/Linux:  source .venv/bin/activate
pip install scikit-learn pandas matplotlib joblib
pip freeze > requirements.txt
printf ".venv/\n__pycache__/\n*.pkl\n" > .gitignore
```

I selected the `.venv` interpreter in VS Code (`Ctrl+Shift+P` -> Python: Select Interpreter).

### Step 2: Wrote `train.py`

It does the full ML pipeline:

1. Loads the Iris data into a pandas table.
2. Separates the inputs `X` (4 measurements) from the answer `y` (species).
3. Splits the data: 80% for training, 20% for testing. The test flowers are never shown to the model while it learns.
4. Trains a **K-Nearest Neighbors (KNN)** model with `fit`.
5. Tests it on the unseen 20% with `predict`, and prints the accuracy and a confusion matrix.
6. Saves a chart (`plot.png`) and the trained model (`model.pkl`).

Run it with `python train.py`.

### Step 3: Wrote `predict.py`

Loads the saved `model.pkl` and predicts the species of one new flower, without training again.

Run it with `python predict.py`.

### Step 4: Wrote `experiments.py`

A script that repeats the train-and-test steps many times, changing **only one thing each time**, and records the accuracy. It then saves:

- `experiments.csv`: every result, one row per run
- `experiments.png`: one chart per experiment

Run it with `python experiments.py`.

| Experiment | What changes | Question it answers |
|---|---|---|
| `k` | number of neighbors (1 to 50) | Is a small or a large k better? |
| `test_size` | share of data held out for testing | Does the test set size change the result? |
| `train_fraction` | how much training data the model sees | Does more data help? |
| `noise` | random noise added to the measurements | How well does the model handle messy data? |
| `random_state` | which random split is used | How much can one lucky or unlucky split mislead me? |
| `features` | which measurements are used | Which measurements matter most? |
| `model` | KNN, decision tree, logistic regression | Does the model choice matter here? |
| `shuffled_labels` | training answers scrambled | Is the model really learning, or guessing? |

### Step 5: Saved my work with Git

```bash
git add .
git commit -m "Baseline model, predict script and experiments"
git log --oneline
```

I commit after each change, so the history shows what I tried.

## 3. Key ideas I learned

- **Train/test split:** a model must be judged on data it has not seen. A score on training data means nothing.
- **KNN:** to classify a new flower, find the k closest flowers in the training data and take a vote. `fit` only stores the data. The work happens in `predict`.
- **Overfitting:** memorizing the training data instead of learning the general pattern. A very small k can do this.
- **Accuracy is not enough:** with unbalanced classes, a model can have high accuracy and still miss the important cases. Precision, recall and F1 give a fuller picture.
- **One split can mislead:** with a test set of 30 flowers, each mistake moves the score by about 3.3%.
- **Reproducibility:** `random_state`, `requirements.txt` and Git commits let me get the same results again.

## 4. What happened (my results)

> Fill this in after running `python experiments.py`. Use `experiments.csv` and `experiments.png`.

**Baseline** (`train.py`, k=5, test_size=0.2, random_state=42): accuracy = ______

| Experiment | Best setting | Best accuracy | Worst setting | Worst accuracy | What I noticed |
|---|---|---|---|---|---|
| k | | | | | |
| test_size | | | | | |
| train_fraction | | | | | |
| noise | | | | | |
| random_state | | | | | |
| features | | | | | |
| model | | | | | |
| shuffled_labels | | | | | |

**Things to check against my results:**

- Did accuracy get worse when k was very small or very large?
- Did accuracy rise as the model got more training data?
- Did accuracy fall as noise increased?
- How different were the 10 `random_state` scores from each other?
- Were the petal measurements enough on their own?
- Did the shuffled labels drop accuracy to about 33% (guessing)? If yes, the model was really learning from the measurements.

**My conclusions:**

1. ______
2. ______
3. ______

## 5. Project files

```
iris-ml/
├── train.py              # train + evaluate + save the model
├── predict.py            # use the saved model on a new flower
├── experiments.py        # run all experiments, save results
├── experiments.csv       # results of every run (generated)
├── experiments.png       # charts of the results (generated)
├── plot.png              # petal length vs width (generated)
├── model.pkl             # trained model (generated, ignored by git)
├── requirements.txt      # package versions
└── README.md
```

## 6. What I want to learn next

1. Explore data with pandas (`.describe()`, `.info()`, missing values).
2. Feature scaling and scikit-learn `Pipeline`.
3. Cross-validation and `GridSearchCV`.
4. A regression project (California housing) and a messier real dataset (Titanic).

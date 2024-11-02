import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import LogisticRegression


def main():
    df = pd.read_csv("diabetes_clean.csv")
    y = df['diabetes']
    X = df.drop(columns=['diabetes'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    params = {"penalty": ['l1', 'l2'],
              "tol": np.linspace(0.0001, 1, 50),
              "C": np.linspace(0.1, 1, 50),
              "class_weight": ["balanced ", {0: 0.8, 1: 0.2}]}
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    log_reg = LogisticRegression()

    gs = RandomizedSearchCV(log_reg, params, cv=kf, random_state=42)
    gs.fit(X_train, y_train)

    print("Tuned Logistic Regression Parameters:", gs.best_params_)
    print("Tuned Logistic Regression Best Accuracy Score:", gs.best_score_)


if __name__ == "__main__":
    main()

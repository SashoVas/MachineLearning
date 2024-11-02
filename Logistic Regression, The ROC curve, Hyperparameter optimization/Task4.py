import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import numpy as np


def main():
    df = pd.read_csv("diabetes_clean.csv")
    y = df['glucose']
    X = df.drop(columns=['glucose'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    kf = KFold(6, random_state=42, shuffle=True)

    model = Lasso()
    gs = GridSearchCV(model, {"alpha": np.linspace(0.00001, 1, 20)}, cv=kf)
    gs.fit(X_train, y_train)

    print("Tuned lasso paramaters:", gs.best_params_)
    print("Tuned lasso score:", gs.best_score_)
    print("Използването на оптимални параметри не гарантира добро представяне на модела. Тук проблема е в самия модел, а не в неговите хиперпараметри.")


if __name__ == "__main__":
    main()

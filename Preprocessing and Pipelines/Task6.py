import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline


def main():
    df = pd.read_csv('music_clean.csv', index_col=0)
    y = df['genre']
    X = df.drop(columns=['genre'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21, stratify=y)
    steps1 = [
        ('LogReg', LogisticRegression(random_state=21))
    ]
    steps2 = [
        ('scaler', StandardScaler()),
        ('LogReg', LogisticRegression(random_state=21))
    ]

    pipeline1 = Pipeline(steps1)
    pipeline2 = Pipeline(steps2)
    params = {'LogReg__C': np.linspace(0.001, 1, 20)}
    kf = KFold(n_splits=6, shuffle=True, random_state=21)

    gs1 = GridSearchCV(pipeline1, params, cv=kf)
    gs1.fit(X_train, y_train)
    print('Without scaling:', gs1.score(X_test, y_test))
    print('Without scaling:', gs1.best_params_)

    gs2 = GridSearchCV(pipeline2, params, cv=kf)
    gs2.fit(X_train, y_train)
    print('With scaling:', gs2.score(X_test, y_test))
    print('With scaling:', gs2.best_params_)
    print("We can see that there is, a big difference in the accuracies of the two models. The one with the standardization have 10% more accuracy, which means that the standardization really helps")


if __name__ == '__main__':
    main()

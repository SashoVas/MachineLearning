import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def main():
    df = pd.read_json('music_dirty_missing_vals.txt')

    df = df.dropna(subset=['genre'])

    # counts = df.isna().mean().sort_values(ascending=False)
    # columns_to_drop = counts[counts < 0.05].index.to_list()
    # df = df.dropna(subset=columns_to_drop)

    y = df['genre']
    X = df.drop(columns=['genre'])
    y = (df['genre'] == 'Rock').astype('int32')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('Logreg', LogisticRegression())
    ])
    params = {'Logreg__solver': ['newton-cg', 'saga', 'lbfgs'],
              'Logreg__C': np.linspace(0.001, 1, 10, endpoint=True)}
    gs = GridSearchCV(pipeline, params)
    gs.fit(X_train, y_train)
    print('Tuned Logistic Regression Parameters:', gs.best_params_)
    print('Accuracy:', gs.best_score_)


if __name__ == "__main__":
    main()

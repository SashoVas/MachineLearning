import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    df = pd.read_csv('music_clean.csv', index_col=0)
    y = df['loudness']
    X = df.drop(columns=['loudness'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    print(X_train.head())
    preprocessing_steps = [
        ('scale', StandardScaler()),
        ('lasso', Lasso(alpha=0.5, random_state=42))
    ]
    no_preprocessing_steps = [
        ('lasso', Lasso(alpha=0.5, random_state=42))
    ]
    pipeline = Pipeline(preprocessing_steps)
    pipeline2 = Pipeline(no_preprocessing_steps)
    pipeline2.fit(X_train, y_train)
    pipeline.fit(X_train, y_train)
    print('Without scaling:', pipeline2.score(X_test, y_test))
    print('With scaling:', pipeline.score(X_test, y_test))


if __name__ == "__main__":
    main()

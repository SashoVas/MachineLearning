import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("diabetes_clean.csv")
    y = df['diabetes']
    X = df.drop(columns=['diabetes'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    results = model.predict_proba(X_test[:10])[:, 1]
    print(results)


if __name__ == "__main__":
    main()

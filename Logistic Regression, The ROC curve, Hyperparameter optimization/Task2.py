import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("diabetes_clean.csv")
    y = df['diabetes']
    X = df.drop(columns=['diabetes'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # metrics.RocCurveDisplay.from_predictions(
    #    y_test, model.predict_proba(X_test)[:, 1])

    fpr, tpr, thresholds = metrics.roc_curve(
        y_test, model.predict_proba(X_test)[:, 1])

    plt.plot([0, 1], [0, 1], 'r')
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.suptitle("ROC curve")
    plt.plot(fpr, tpr)
    plt.tight_layout()
    plt.show()
    print("C. The model is much better than randomly guessing the class of each observation.")


if __name__ == "__main__":
    main()

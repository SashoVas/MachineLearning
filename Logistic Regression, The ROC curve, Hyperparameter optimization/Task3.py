import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    df = pd.read_csv("diabetes_clean.csv")
    y = df['diabetes']
    X = df.drop(columns=['diabetes'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print("Model KNN trained!")
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    print("Model LogisticRegression trained!")

    predictions_proba = knn.predict_proba(X_test)[:, 1]
    predictions = knn.predict(X_test)
    print("KNN AUC:", roc_auc_score(y_test, predictions_proba))
    print("KNN Metrics:", classification_report(y_test, predictions))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax0)

    predictions_proba = log_reg.predict_proba(X_test)[:, 1]
    predictions = log_reg.predict(X_test)
    print("LogisticRegression AUC:", roc_auc_score(y_test, predictions_proba))
    print("LogisticRegression Metrics:",
          classification_report(y_test, predictions))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax1)

    print('Logistic regression е моделът, който се справя по-добре. Вижда се по всички харакетеристика като auc, f1 и всички други.')
    plt.show()


if __name__ == "__main__":
    main()

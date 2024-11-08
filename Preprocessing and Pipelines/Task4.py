import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    df = pd.read_json('music_dirty_missing_vals.txt')
    # df = df.dropna(subset=['genre'])

    counts = df.isna().mean().sort_values(ascending=False)
    columns_to_drop = counts[counts < 0.05].index.to_list()
    # df = df.dropna(subset=columns_to_drop)
    df['genre'] = (df['genre'] == 'Rock').astype('int32')

    y = df['genre']
    X = df.drop(columns=['genre'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('knn', KNeighborsClassifier(n_neighbors=3))

    ]
    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)
    results = pipeline.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=results))
    print("""We can see that our model does not perform very good on the data.
We can deduce that based on all the metrics. 
For example the auc score is 0.57,
which is a little better than a model predicting at random(expected auc of random prediction is around 0.5).
The other metrics like precision, recall and f1-score, indicate that the predictions are a little better than random.
          """)
    RocCurveDisplay.from_predictions(y_true=y_test, y_pred=results)
    ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=results)
    plt.show()


if __name__ == "__main__":
    main()

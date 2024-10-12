import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


class KNaerestNeighbours:

    def __init__(self, k=1):
        self.k = k

    def fit(self, train_x, train_y):
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)

    def predict(self, point):

        if self.train_x is None or self.train_y is None:
            print('No training data provided')
            return

        axis = 1 if len(self.train_x.shape) != 1 else 0
        # It is not needed to take the sqrt of the distance, because it does not change the order in the sorted array
        distances = np.sum((self.train_x - point)**2, axis=axis)

        k_naerest_neighbours_labels = np.argsort(distances)[:self.k]
        return stats.mode(self.train_y[k_naerest_neighbours_labels]).mode

    def score(self, test_x, test_y):
        test_res = np.apply_along_axis(self.predict, 1, test_x)
        correct_predictions = test_res == test_y
        return np.sum(correct_predictions)/len(correct_predictions)


def test():
    df = pd.read_csv('telecom_churn_clean.csv', index_col=0)
    y = np.array(df['churn'])
    X = np.array(df.drop(columns=['churn']))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print('Accuracy sklearn with "n_neighbors=5":', knn.score(X_test, y_test))

    knn2 = KNaerestNeighbours(k=5)
    knn2.fit(X_train, y_train)
    print('Accuracy custom implementation with "n_neighbors=5":',
          knn2.score(X_test, y_test))

    neighbors = np.arange(1, 13)

    for n_neighbors in neighbors:

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        knn2 = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn2.fit(X_train, y_train)

        if knn.score(X_test, y_test) != knn2.score(X_test, y_test):
            print('Different result')
            return

    print("They are equivelent for every k value in the range [1,12]")


if __name__ == '__main__':
    test()

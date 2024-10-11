import numpy as np
from scipy import stats


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


if __name__ == 'main':
    knn = KNaerestNeighbours(2)
    train_x = np.array([[14, 2], [2, 4], [43, 1], [34, 12],
                        [5, 3], [26, 0], [74, 56], [0, 0]])
    train_y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    knn.fit(train_x, train_y)

    print(knn.score(train_x, train_y))

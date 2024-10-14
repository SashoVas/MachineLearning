import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('telecom_churn_clean.csv', index_col=0)
    y = np.array(df['churn'])
    X = np.array(df.drop(columns=['churn']))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print('Training Dataset Shape:', X_train.shape)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print('Accuracy when "n_neighbors=5":', knn.score(X_test, y_test))

    train_acc = {}
    test_acc = {}
    neighbors = np.arange(1, 76)

    for n_neighbors in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        train_acc[int(n_neighbors)] = knn.score(X_train, y_train)
        test_acc[int(n_neighbors)] = knn.score(X_test, y_test)

    print(f'neighbors={neighbors}')
    print(f'train_accuracies={train_acc}')
    print(f'test_accuracies={test_acc}')

    max_point = max(test_acc.items(), key=lambda x: x[1])
    print(max_point)
    plt.suptitle('KNN: Varying Number of Neighbors')
    plt.xlabel('Number of Neighbours')
    plt.ylabel('Accuracy')
    plt.plot(train_acc.keys(), train_acc.values(), label='Training accuracy')
    plt.plot(test_acc.keys(), test_acc.values(), label='Testing accuracy')
    plt.scatter(max_point[0], max_point[1], linewidths=0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

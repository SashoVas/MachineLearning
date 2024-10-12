import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


def main():
    df = pd.read_csv('telecom_churn_clean.csv', index_col=0)
    y_train = np.array(df['churn'])
    X_train = np.array(df[['account_length', 'customer_service_calls']])

    knn = KNeighborsClassifier(n_neighbors=6)

    knn.fit(X_train, y_train)

    X_new = np.array([[30.0, 17.5],
                      [107.0, 24.1],
                      [213.0, 10.9]])

    print(knn.predict(X_new))


if __name__ == '__main__':
    main()

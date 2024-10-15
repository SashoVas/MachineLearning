import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


if __name__ == '__main__':

    df = pd.read_csv('advertising_and_sales_clean.csv')
    df = df.drop(columns=['influencer'])
    y = df['sales']
    # you can use df.drop, but this does not make a copy
    X = df.loc[:, df.columns != 'sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print('Predictions:', model.predict(X_test[:2]))
    print('Actual Values:', y_test[:2].values)

    print(model.score(X_test, y_test))
    print(root_mean_squared_error(model.predict(X_test), y_test))

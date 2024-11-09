import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline


def main():

    df = pd.read_csv('music_clean.csv', index_col=0)
    y = df['energy']
    X = df.drop(columns=['energy'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    lin_reg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('Linear regression', LinearRegression())

    ])

    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('Ridge', Ridge(alpha=0.1))

    ])

    lin_reg_pipeline.fit(X_train, y_train)
    ridge_pipeline.fit(X_train, y_train)

    print('Linear Regression Test Set RMSE:', root_mean_squared_error(
        y_test, lin_reg_pipeline.predict(X_test)))
    print('Ridge Test Set RMSE:', root_mean_squared_error(
        y_test, ridge_pipeline.predict(X_test)))


if __name__ == "__main__":
    main()

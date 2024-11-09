import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def main():
    df = pd.read_csv('music_clean.csv', index_col=0)
    y = df['energy']
    X = df.drop(columns=['energy'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    model1 = LinearRegression()
    model2 = Ridge(alpha=0.1)
    model3 = Lasso(alpha=0.1)
    model1_res = cross_val_score(model1, X_train, y_train, cv=kf)
    model2_res = cross_val_score(model2, X_train, y_train, cv=kf)
    model3_res = cross_val_score(model3, X_train, y_train, cv=kf)
    print('We can see that the Linear regression and ridge models perform better than the lasso. The linear regression and ridge yield fairly similar results.')
    plt.boxplot([model1_res, model2_res, model3_res], tick_labels=[
                'Linear Regression', 'Ridge', 'Lasso'])
    plt.show()


if __name__ == "__main__":
    main()

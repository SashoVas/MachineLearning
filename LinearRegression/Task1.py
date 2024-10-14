import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def plot_reletions(df):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.scatter(df['radio'], df['sales'])
    ax1.set_title('radio vs sales')
    ax1.set_xlabel('Radio')
    ax1.set_ylabel('Sales')

    ax2.scatter(df['tv'], df['sales'])
    ax2.set_title('tv vs sales')
    ax2.set_xlabel('TV')
    ax2.set_ylabel('Sales')

    ax3.scatter(df['social_media'], df['sales'])
    ax3.set_title('social_media vs sales')
    ax1.set_xlabel('Social Media')
    ax1.set_ylabel('Sales')

    plt.tight_layout()
    plt.show()


def plot_result(model, X, y):
    plt.scatter(X, y)
    plt.plot(X, model.predict(X), 'r')
    plt.suptitle('Relationship between radio expenditures and sales')
    plt.ylabel('Sales ($)')
    plt.xlabel('Radio Expenditure ($)')
    plt.show()


if __name__ == '__main__':

    df = pd.read_csv('advertising_and_sales_clean.csv')
    print(df.head(2))
    df = df.drop(columns=['influencer'])
    print(df.corr()['sales'])

    plot_reletions(df)

    y_train = df['sales']
    X_train = df[['radio']]
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.predict(X_train[:5]))

    plot_result(model, X_train, y_train)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


def main():

    df = pd.read_csv('music_clean.csv', index_col=0)
    y = df['popularity']
    X = df.drop(columns=['popularity'])
    y = (y >= y.median()).astype('int32')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    kf = KFold(n_splits=6, shuffle=True, random_state=12)

    model1 = Pipeline([
        ('scaler', StandardScaler()),
        ('log reg', LogisticRegression(random_state=42))])

    model2 = Pipeline([
        ('scaler', StandardScaler()),
        ('log reg', KNeighborsClassifier(n_neighbors=6))])

    model3 = Pipeline([
        ('scaler', StandardScaler()),
        ('log reg', DecisionTreeClassifier(random_state=42))])

    res1 = cross_val_score(model1, X_train, y_train, cv=kf)
    res2 = cross_val_score(model2, X_train, y_train, cv=kf)
    res3 = cross_val_score(model3, X_train, y_train, cv=kf)
    print('The model that performs best is logistic regression in terms of accuracy . It clearly outperforms the DecisionTreeClassifier. It is a little better than knn. The 25% of the box plot is similar with the knn, but the 75% is higher, and the median is higher too.')
    plt.boxplot([res1, res2, res3], tick_labels=[
                'Logistic Regression', 'KNN', 'Decision Tree Classifier'])
    plt.show()


if __name__ == "__main__":
    main()

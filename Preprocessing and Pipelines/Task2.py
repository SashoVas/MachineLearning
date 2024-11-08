import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
import numpy as np


def main():
    df = pd.read_json('music_dirty.txt')
    genres = pd.get_dummies(df['genre'], drop_first=True)
    df = pd.concat([df.drop(columns=['genre']), genres], axis=1)
    y = df['popularity']
    X = df.drop(columns=['popularity'])

    model = Ridge(alpha=0.2)
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    result = -cross_val_score(
        model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    print('Average RMSE:', np.mean(result))
    print('Standard Deviation of the target array:', y.std())
    print("From what we see, we can conclude that our model performs much better than the mean, because the std is calculated based on the mean.")


if __name__ == "__main__":
    main()

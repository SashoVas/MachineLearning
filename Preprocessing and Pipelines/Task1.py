import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_json('music_dirty.txt')
    genres = pd.get_dummies(df['genre'], drop_first=True)
    print("Shape before one-hot-encoding:", df.shape)
    df = pd.concat([df.drop(columns=['genre']), genres], axis=1)
    print("Shape after one-hot-encoding:", df.shape)
    data = [df[df[genre]]['popularity'] for genre in genres.columns]
    plt.boxplot(data, tick_labels=genres.columns)
    plt.suptitle("Boxplot grouped by genre")
    plt.xlabel("genre")
    plt.ylabel("popularity")
    plt.show()


if __name__ == "__main__":
    main()

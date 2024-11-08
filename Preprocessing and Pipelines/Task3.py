import pandas as pd
import numpy as np


def main():
    df = pd.read_json('music_dirty_missing_vals.txt')
    print('Shape of input dataframe:', df.shape)
    print("Percentage of missing values:")
    counts = df.isna().mean().sort_values(ascending=False)
    print(counts)
    columns_to_drop = counts[counts < 0.05].index.to_list()
    df = df.dropna(subset=columns_to_drop)
    print('Columns/Variables with missing values less than 5% of the dataset:', columns_to_drop)
    df['genre'] = (df['genre'] == 'Rock').astype('int32')
    print(df['genre'].head())
    print('Shape of preprocessed dataframe:', df.shape)


if __name__ == "__main__":
    main()

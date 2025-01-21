import pandas as pd
from sklearn.datasets import load_digits
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def plot_images(digits):
    fig = plt.figure(figsize=(8, 2))
    fig.subplots_adjust(left=0, right=1, bottom=0,
                        top=1, hspace=0.1, wspace=0.1)

    i = 0
    for img in digits.images[0:6]:
        ax = fig.add_subplot(1, 6, i + 1, xticks=[], yticks=[])
        ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
        i += 1

    plt.show()


def main():
    digits = load_digits()
    print('Dataset shape: ', digits.data.shape)
    df = pd.DataFrame(np.column_stack(
        [digits['data'], digits['target']]), columns=digits['feature_names'] + ['target'])
    print('Number of classes: ', len(df['target'].unique()))
    y = df['target']
    X = df.drop(columns=['target'])
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=42, random_state=42, stratify=y)
    log_reg = LogisticRegression()
    log_reg.fit(train_x, train_y)
    print('Training accuracy of logistic regression:',
          log_reg.score(train_x, train_y))
    print('Validation accuracy of logistic regression:',
          log_reg.score(test_x, test_y))

    support_vector_classifier = SVC()
    support_vector_classifier.fit(train_x, train_y)
    print('Training accuracy of non-linear support vector classifier:',
          support_vector_classifier.score(train_x, train_y))
    print('Validation accuracy of non-linear support vector classifier:',
          support_vector_classifier.score(test_x, test_y))
    plot_images(digits)
    print('The SVC is obviously the better classifier because of the validation acc')


if __name__ == '__main__':
    main()

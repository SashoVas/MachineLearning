## Logistic Regression

### Introduction

It's time to introduce another model: `Logistic Regression`.

<details>

<summary>Is logistic regression used for regression or classification problem?</summary>

Despite its name, **logistic regression is used for solving classification problems**.

This model:

- calculates the probability - `p`, that an observation belongs to a class;
- is usually used in binary classification, though it's also popular in multi-class classification;
- produces a linear decision boundary.

![w04_log_reg_boundary.png](./assets/w04_log_reg_boundary.png "w04_log_reg_boundary.png")

Using our diabetes dataset as an example:

- if `p >= 0.5`, we label the data as `1`, representing a prediction that an individual is more likely to have diabetes;
- if `p < 0.5`, we label it `0` to represent that they are more likely to not have diabetes.

</details>

<details>

<summary>Have you heard about the iris dataset?</summary>

- The [`iris` dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) is a collection of measurements of many [iris plants](https://en.wikipedia.org/wiki/Iris_flower_data_set).
- Three species of iris:
  - *setosa*;
  - *versicolor*;
  - *virginica*.
- Features: petal length, petal width, sepal length, sepal width.

Iris setosa:

![w05_setosa.png](./assets/w05_setosa.png "w05_setosa.png")

Iris versicolor:

![w05_versicolor.png](./assets/w05_versicolor.png "w05_versicolor.png")

Iris virginica:

![w05_virginica.png](./assets/w05_virginica.png "w05_virginica.png")

</details>

### In `scikit-learn`

In scikit-learn logistic regression is implemented in the [`sklearn.linear_model`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#logisticregression) module:

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
print(f'Possible classes: {set(y)}')
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
```

```console
Possible classes: {0, 1, 2}
array([0, 0])
```

> **Note:** The hyperparameter `C` is the inverse of the regularization strength - larger `C` means less regularization and smaller `C` means more regularization.

```python
clf.predict_proba(X[:2]) # probabilities of each instance belonging to each of the three classes
```

```console
array([[9.81780846e-01, 1.82191393e-02, 1.44184120e-08],
       [9.71698953e-01, 2.83010167e-02, 3.01417036e-08]])
```

```python
clf.predict_proba(X[:2])[:, 2] # probabilities of each instance belonging to the third class only
```

```console
array([1.44184120e-08, 3.01417036e-08])
```

```python
clf.score(X, y) # returns the accuracy
```

```console
0.97
```

## The receiver operating characteristic curve (`ROC` curve)

<details>

<summary>Do you know what the ROC curve shows?</summary>

> **Note:** Use the `ROC` curve only when doing ***binary* classification**.

The default probability threshold for logistic regression in scikit-learn is `0.5`. What happens as we vary this threshold?

We can use a receiver operating characteristic, or ROC curve, to visualize how different thresholds affect `true positive` and `false positive` rates.

![w04_roc_example.png](./assets/w04_roc_example.png "w04_roc_example.png")

The dotted line represents a random model - one that randomly guesses labels.

When the threshold:

- equals `0` (`p=0`), the model predicts `1` for all observations, meaning it will correctly predict all positive values, and incorrectly predict all negative values;
- equals `1` (`p=1`), the model predicts `0` for all data, which means that both true and false positive rates are `0` (nothing gets predicted as positive).

![w04_roc_edges.png](./assets/w04_roc_edges.png "w04_roc_edges.png")

If we vary the threshold, we get a series of different false positive and true positive rates.

![w04_roc_vary.png](./assets/w04_roc_vary.png "w04_roc_vary.png")

A line plot of the thresholds helps to visualize the trend.

![w04_roc_line.png](./assets/w04_roc_line.png "w04_roc_line.png")

<details>

<summary>What plot would be produced by the perfect model?</summary>

One in which the line goes straight up and then right.

![w04_perfect_roc.png](./assets/w04_perfect_roc.png "w04_perfect_roc.png")

</details>

</details>

### In `scikit-learn`

In scikit-learn the `roc_curve` is implemented in the `sklearn.metrics` module.

```python
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

fpr
```

```console
array([0. , 0. , 0.5, 0.5, 1. ])
```

```python
tpr
```

```console
array([0. , 0.5, 0.5, 1. , 1. ])
```

```python
thresholds
```

```console
array([ inf, 0.8 , 0.4 , 0.35, 0.1 ])
```

To plot the curve can create a `matplotlib` plot or use a built-in function.

Using `matplotlib` would look like this:

```python
plt.plot([0, 1], [0, 1], 'k--') # to draw the dashed line
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

![w04_example_roc_matlotlib.png](./assets/w04_example_roc_matlotlib.png "w04_example_roc_matlotlib.png")

We could also use the `from_predictions` function in the `RocCurveDisplay` class to create plots.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y, scores)
plt.tight_layout()
plt.show()
```

![w04_example_roc_display.png](./assets/w04_example_roc_display.png "w04_example_roc_display.png")

<details>

<summary>The above figures look great, but how do we quantify the model's performance based on them?</summary>

### The Area Under the Curve (`AUC`)

If we have a model with `true_positive_rate=1` and `false_positive_rate=0`, this would be the perfect model.

Therefore, we calculate the area under the ROC curve, a **metric** known as `AUC`. Scores range from `0` to `1`, with `1` being ideal.

In the below figure, the model scores `0.67`, which is `34%` better than a model making random guesses.

![w04_example_roc_improvement.png](./assets/w04_example_roc_improvement.png "w04_example_roc_improvement.png")

### In `scikit-learn`

In scikit-learn the area under the curve can be calculated in two ways.

Either by using the `RocCurveDisplay.from_predictions` function:

![w04_example_roc_display_note.png](./assets/w04_example_roc_display_note.png "w04_example_roc_display_note.png")

or by using the `roc_auc_score` function in the `sklearn.metrics` module:

```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```

```console
0.6700964152663693
```

</details>

## Hyperparameter optimization/tuning

### Introduction

Recall that we had to choose a value for `alpha` in ridge and lasso regression before fitting it.

Likewise, before fitting and predicting `KNN`, we choose `n_neighbors`.

> **Definition:** Parameters that we specify before fitting a model, like `alpha` and `n_neighbors`, are called **hyperparameters**.

<details>

<summary>So, a fundamental step for building a successful model is choosing the correct hyperparameters. What's the best way to go about achieving this?</summary>

We can try lots of different values, fit all of them separately, see how well they perform, and choose the best values!

> **Definition:** The process of trying out different hyperparameters until a satisfactory performance threshold is reached is called **hyperparameter tuning**.

</details>

When fitting different hyperparameter values, we use cross-validation to avoid overfitting the hyperparameters to the test set:

- we split the data into train and test;
- and perform cross-validation on the training set.

We withhold the test set and use it only for evaluating the final, tuned model.

> **Definition:** The set on which the model is evaluated on during cross-validation is called the **validation set**.

Notice how:

- the training set is used to train the model;
- the validation set is used to tune the model until a satisfactory performance threshold is used;
- the test set is used as the final set on which model performance is reported. It is data that the model hasn't seen, but because it is labeled, we can use to evaluate the performance eliminating bias.

### Grid search cross-validation

<details>

<summary>Do you know what grid search cross-validation comprises of?</summary>

One approach for hyperparameter tuning is called **grid search**, where we choose a grid of possible hyperparameter values to try. In Python this translates to having a dictionary that maps strings to lists/arrays of possible values to choose from.

For example, we can search across two hyperparameters for a `KNN` model - the type of metric and a different number of neighbors. We perform `k`-fold cross-validation for each combination of hyperparameters. The mean scores (in this case, accuracies) for each combination are shown here:

![w04_grid_search_cv.png](./assets/w04_grid_search_cv.png "w04_grid_search_cv.png")

We then choose hyperparameters that performed best:

![w04_grid_search_cv_choose_best.png](./assets/w04_grid_search_cv_choose_best.png "w04_grid_search_cv_choose_best.png")

To get a list of supported values for the model we're building, we can use the scikit-learn documentation. For example, in the documentation for [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#logisticregression), the possible values for the `solver` parameter can be seen as one scrolls down the page.

### In `scikit-learn`

`scikit-learn` has an implementation of grid search using cross-validation in the `sklearn.model_selection` module:

```python
from sklearn.model_selection import GridSearchCV

# instantiate a `KFold` object

param_grid = {
    'alpha': np.arange(0.0001, 1, 10),
    'solver': ['sag', 'lsqr']
}

ridge_cv = GridSearchCV(Ridge(), param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

```console
{'alpha': 0.0001, 'solver': 'sag'}
0.7529912278705785
```

</details>

<details>

<summary>What is the main problem of grid search?</summary>

Grid search is great - it allows us to scan a predefined parameter space fully. However, it does not scale well:

<details>

<summary>How many fits will be done while performing 3-fold cross-validation for 1 hyperparameter with 10 values?</summary>

Answer: 30.

</details>

<details>

<summary>How many fits will be done while performing 10-fold cross-validation for 3 hyperparameters with 10 values each?</summary>

Answer: 10,000!

We can verify this:

```python
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


def main():
    iris = datasets.load_iris()

    parameters = {
        'degree': np.arange(10),
        'C': np.linspace(0, 10, 10),
        'tol': np.linspace(0.0001, 0.01, 10),
    }

    print(len(parameters['degree']))
    print(len(parameters['C']))
    print(len(parameters['tol']))

    svc = svm.SVC() # This is a support vector machine. We'll talk about it soon.
    clf = GridSearchCV(svc, parameters, cv=10, verbose=1)
    clf.fit(iris.data, iris.target)
    print(sorted(clf.cv_results_))


if __name__ == '__main__':
    main()
```

This is because:

1. The total number of parameter combinations is `10^3 = 1000` (we have one for-loop with two nested ones inside).

  ```python
  # pseudocode
  for d in degrees:
    for c in cs:
      for tol in tols:
        # this is one combination
  ```

2. For every single one combination we do a `10`-fold cross-validation to get the mean metric. This means that every single one of the paramter combinations is the same while we shift the trainig and testing sets `10` times.

</details>

<details>

<summary>What is the formula in general then?</summary>

```text
number of fits = number of folds * number of total hyperparameter values
```

</details>

<details>

<summary>How can we go about solving this?</summary>

### Randomized search cross-validation

We can perform a random search, which picks random hyperparameter values rather than exhaustively searching through all options.

```python
from sklearn.model_selection import RandomizedSearchCV

# instantiate a `KFold` object

param_grid = {
    'alpha': np.arange(0.0001, 1, 10),
    'solver': ['sag', 'lsqr']
}

# optionally set the "n_iter" argument, which determines the number of hyperparameter values tested (default is 10)
# 5-fold cross-validation with "n_iter" set to 2 performs 10 fits
ridge_cv = RandomizedSearchCV(Ridge(), param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

In this case it is able to find the best hyperparameters from our previous grid search!

```console
{'alpha': 0.0001, 'solver': 'sag'}
0.7529912278705785
```

#### Benefits

This allows us to search from large parameter spaces efficiently.

### Evaluating on the test set

We can evaluate model performance on the test set by passing it to a call of the grid/random search object's `.score` method.

```python
test_score = ridge_cv.score(X_test, y_test)
test_score
```

```console
0.7564731534089224
```

</details>

</details>
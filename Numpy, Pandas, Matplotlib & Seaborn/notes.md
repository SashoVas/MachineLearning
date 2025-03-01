# Week 01 - Numpy, Pandas, Matplotlib & Seaborn

## Python Packages

### Introduction

You write all of your code to one and the same Python script.

<details>

<summary>What are the problems that arise from that?</summary>

- Huge code base: messy;
- Lots of code you won't use;
- Maintenance problems.

</details>

<details>

<summary>How do we solve this problem?</summary>

We can split our code into libraries (or in the Python world - **packages**).

Packages are a directory of Python scripts.

Each such script is a so-called **module**.

Here's the hierarchy visualized:

![w01_packages_modules.png](./assets/w01_packages_modules.png "w01_packages_modules.png")

These modules specify functions, methods and new Python types aimed at solving particular problems. There are thousands of Python packages available from the Internet. Among them are packages for data science:

- there's **NumPy to efficiently work with arrays**;
- **Matplotlib for data visualization**;
- **scikit-learn for machine learning**.

</details>

Not all of them are available in Python by default, though. To use Python packages, you'll first have to install them on your own system, and then put code in your script to tell Python that you want to use these packages. Advice:

- always install packages in **virtual environments** (abstractions that hold packages for separate projects).
  - You can create a virtual environment by using the following code:

    ```console
    python3 -m venv .venv
    ```

    This will create a hidden folder, called `.venv`, that will store all packages you install for your current project (instead of installing them globally on your system).

  - If there is a `requirements.txt` file, use it to install the needed packages beforehand.
    - In the github repo, there is such a file - you can use it to install all the packages you'll need in the course. This can be done by using this command:

    ```console
    (if on Windows) > .venv\Scripts\activate
    (if on Linux) > source .venv/bin/activate
    (.venv) > pip install -r requirements.txt
    ```

Now that the package is installed, you can actually start using it in one of your Python scripts. To do this you should import the package, or a specific module of the package.

You can do this with the `import` statement. To import the entire `numpy` package, you can do `import numpy`. A commonly used function in NumPy is `array`. It takes a Python list as input and returns a [`NumPy array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) object as an output. The NumPy array is very useful to do data science, but more on that later. Calling the `array` function like this, though, will generate an error:

```python
import numpy
array([1, 2, 3])
```

```console
NameError: name `array` is not defined
```

To refer to the `array` function from the `numpy` package, you'll need this:

```python
import numpy
numpy.array([1, 2, 3])
```

```console
array([1, 2, 3])
```

This time it works.

Using this `numpy.` prefix all the time can become pretty tiring, so you can also import the package and refer to it with a different name. You can do this by extending your `import` statement with `as`:

```python
import numpy as np
np.array([1, 2, 3])
```

```console
array([1, 2, 3])
```

Now, instead of `numpy.array`, you'll have to use `np.array` to use NumPy's functions.

There are cases in which you only need one specific function of a package. Python allows you to make this explicit in your code.

Suppose that we ***only*** want to use the `array` function from the NumPy package. Instead of doing `import numpy`, you can instead do `from numpy import array`:

```python
from numpy import array
array([1, 2, 3])
```

```console
array([1, 2, 3])
```

This time, you can simply call the `array` function without `numpy.`.

This `from import` version to use specific parts of a package can be useful to limit the amount of coding, but you're also loosing some of the context. Suppose you're working in a long Python script. You import the array function from numpy at the very top, and way later, you actually use this array function. Somebody else who's reading your code might have forgotten that this array function is a specific NumPy function; it's not clear from the function call.

![w01_from_numpy.png](./assets/w01_from_numpy.png "w01_from_numpy.png")

^ using numpy, but not very clear

Thus, the more standard `import numpy as np` call is preferred: In this case, your function call is `np.array`, making it very clear that you're working with NumPy.

![w01_import_as_np.png](./assets/w01_import_as_np.png "w01_import_as_np.png")

- Suppose you want to use the function `inv()`, which is in the `linalg` subpackage of the `scipy` package. You want to be able to use this function as follows:

    ```python
    my_inv([[1,2], [3,4]])
    ```

    Which import statement will you need in order to run the above code without an error?

  - A. `import scipy`
  - B. `import scipy.linalg`
  - C. `from scipy.linalg import my_inv`
  - D. `from scipy.linalg import inv as my_inv`

    <details>

    <summary>Reveal answer:</summary>

    Answer: D

    </details>

### Importing means executing `main.py`

Remember that importing a package is equivalent to executing everything in the `main.py` module. Thus. you should always have `if __name__ == '__main__'` block of code and call your functions from there.

Run the scripts `test_script1.py` and `test_script2.py` to see the differences.

## NumPy

### Context

Python lists are pretty powerful:

- they can hold a collection of values with different types (heterogeneous data structure);
- easy to change, add, remove elements;
- many built-in functions and methods.

### Problems

This is wonderful, but one feature is missing, a feature that is super important for aspiring data scientists and machine learning engineers - carrying out mathematical operations **over entire collections of values** and doing it **fast**.

Let's take the heights and weights of your family and yourself. You end up with two lists, `height`, and `weight` - the first person is `1.73` meters tall and weighs `65.4` kilograms and so on.

```python
height = [1.73, 1.68, 1.71, 1.89, 1.79]
height
```

```console
[1.73, 1.68, 1.71, 1.89, 1.79]
```

```python
weight = [65.4, 59.2, 63.6, 88.4, 68.7]
weight
```

```console
[65.4, 59.2, 63.6, 88.4, 68.7]
```

If you now want to calculate the Body Mass Index for each family member, you'd hope that this call can work, making the calculations element-wise. Unfortunately, Python throws an error, because it has no idea how to do calculations on lists. You could solve this by going through each list element one after the other, and calculating the BMI for each person separately, but this is terribly inefficient and tiresome to write.

### Solution

- `NumPy`, or Numeric Python;
- Provides an alternative to the regular Python list: the NumPy array;
- The NumPy array is pretty similar to the list, but has one additional feature: you can perform calculations over entire arrays;
- super-fast as it's based on C++
- Installation:
  - In the terminal: `pip install numpy`

### Benefits

Speed, speed, speed:

- Stackoverflow: <https://stackoverflow.com/questions/73060352/is-numpy-any-faster-than-default-python-when-iterating-over-a-list>
- Visual Comparison:

    ![w01_np_vs_list.png](./assets/w01_np_vs_list.png "w01_np_vs_list.png")

### Usage

```python
import numpy as np
np_height = np.array(height)
np_height
```

```console
array([1.73, 1.68, 1.71, 1.89, 1.79])
```

```python
import numpy as np
np_weight = np.array(weight)
np_weight
```

```console
array([65.4, 59.2, 63.6, 88.4, 68.7])
```

```python
# Calculations are performed element-wise.
# 
# The first person's BMI was calculated by dividing the first element in np_weight
# by the square of the first element in np_height,
# 
# the second person's BMI was calculated with the second height and weight elements, and so on.
bmi = np_weight / np_height ** 2
bmi
```

```console
array([21.851, 20.975, 21.750, 24.747, 21.441])
```

in comparison, the above will not work for Python lists:

```python
weight / height ** 2
```

```console
TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'
```

You should still pay attention, though:

- `numpy` assumes that your array contains values **of a single type**;
- a NumPy array is simply a new kind of Python type, like the `float`, `str` and `list` types. This means that it comes with its own methods, which can behave differently than you'd expect.

    ```python
    python_list = [1, 2, 3]
    numpy_array = np.array([1, 2, 3])
    ```

    ```python
    python_list + python_list
    ```

    ```console
    [1, 2, 3, 1, 2, 3]
    ```

    ```python
    numpy_array + numpy_array
    ```

    ```console
    array([2, 4, 6])
    ```

- When you want to get elements from your array, for example, you can use square brackets as with Python lists. Suppose you want to get the bmi for the second person, so at index `1`. This will do the trick:

    ```python
    bmi
    ```

    ```console
    [21.851, 20.975, 21.750, 24.747, 21.441]
    ```

    ```python
    bmi[1]
    ```

    ```console
    20.975
    ```

- Specifically for NumPy, there's also another way to do list subsetting: using an array of booleans.

    Say you want to get all BMI values in the bmi array that are over `23`.

    A first step is using the greater than sign, like this: The result is a NumPy array containing booleans: `True` if the corresponding bmi is above `23`, `False` if it's below.

    ```python
    bmi > 23
    ```

    ```console
    array([False, False, False, True, False])
    ```

    Next, you can use this boolean array inside square brackets to do subsetting. Only the elements in `bmi` that are above `23`, so for which the corresponding boolean value is `True`, is selected. There's only one BMI that's above `23`, so we end up with a NumPy array with a single value, that specific BMI. Using the result of a comparison to make a selection of your data is a very common way to work with data.

    ```python
    bmi[bmi > 23]
    ```

    ```console
    array([24.747])
    ```

### 2D NumPy Arrays

If you ask for the type of these arrays, Python tells you that they are `numpy.ndarray`:

```python
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
type(np_height)
```

```console
numpy.ndarray
```

```python
type(np_weight)
```

```console
numpy.ndarray
```

`ndarray` stands for n-dimensional array. The arrays `np_height` and `np_weight` are one-dimensional arrays, but it's perfectly possible to create `2`-dimensional, `3`-dimensional and `n`-dimensional arrays.

You can create a 2D numpy array from a regular Python list of lists:

```python
np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79],
                  [65.4, 59.2, 63.6, 88.4, 68.7]])
np_2d
```

```console
array([[ 1.73,  1.68,  1.71,  1.89,  1.79],
       [65.4 , 59.2 , 63.6 , 88.4 , 68.7 ]])
```

Each sublist in the list, corresponds to a row in the `2`-dimensional numpy array. Using `.shape`, you can see that we indeed have `2` rows and `5` columns:

```python
np_2d.shape
```

```console
(2, 5) # 2 rows, 5 columns
```

`shape` is a so-called **attribute** of the `np2d` array, that can give you more information about what the data structure looks like.

> **Note:** The syntax for accessing an attribute looks a bit like calling a method, but they are not the same! Remember that methods have round brackets (`()`) after them, but attributes do not.
>
> **Note:** For n-D arrays, the NumPy rule still applies: an array can only contain a single type.

You can think of the 2D numpy array as a faster-to-work-with list of lists: you can perform calculations and more advanced ways of subsetting.

Suppose you want the first row, and then the third element in that row - you can grab it like this:

```python
np_2d[0][2]
```

```console
1.71
```

or use an alternative way of subsetting, using single square brackets and a comma:

```python
np_2d[0, 2]
```

```console
1.71
```

The value before the comma specifies the row, the value after the comma specifies the column. The intersection of the rows and columns you specified, are returned. This is the syntax that's most popular.

Suppose you want to select the height and weight of the second and third family member from the following array.

```console
array([[ 1.73,  1.68,  1.71,  1.89,  1.79],
       [65.4 , 59.2 , 63.6 , 88.4 , 68.7 ]])
```

<details>

<summary>How can this be achieved?</summary>

Answer: np_2d[:, [1, 2]]

</details>

### Basic Statistics

A typical first step in analyzing your data, is getting to know your data in the first place.

Imagine you conduct a city-wide survey where you ask `5000` adults about their height and weight. You end up with something like this: a 2D numpy array, that has `5000` rows, corresponding to the `5000` people, and `2` columns, corresponding to the height and the weight.

```python
np_city = ...
np_city
```

```console
array([[ 2.01, 64.33],
       [ 1.78, 67.56],
       [ 1.71, 49.04],
       ...,
       [ 1.73, 55.37],
       [ 1.72, 69.73],
       [ 1.85, 66.69]])
```

Simply staring at these numbers, though, won't give you any insights. What you can do is generate summarizing statistics about them.

- you can try to find out the average height of these people, with NumPy's `mean` function:

```python
np.mean(np_city[:, 0]) # alternative: np_city[:, 0].mean()
```

```console
1.7472
```

It appears that on average, people are `1.75` meters tall.

- What about the median height? This is the height of the middle person if you sort all persons from small to tall. Instead of writing complicated python code to figure this out, you can simply use NumPy's `median` function:

```python
np.median(np_city[:, 0]) # alternative: np_city[:, 0].median()
```

```console
1.75
```

You can do similar things for the `weight` column in `np_city`. Often, these summarizing statistics will provide you with a "sanity check" of your data. If you end up with a average weight of `2000` kilograms, your measurements are most likely incorrect. Apart from mean and median, there's also other functions, like:

```python
np.corrcoef(np_city[:, 0], np_city[:, 1])
```

```console
array([[1.       , 0.0082912],
       [0.0082912, 1.       ]])
```

```python
np.std(np_city[:, 0])
```

```console
np.float64(0.19759467357193614)
```

`sum()`, `sort()`, etc, etc. See all of them [here](https://numpy.org/doc/stable/reference/routines.statistics.html).

### Generate data

The data used above was generated using the following code. Two random distributions were sampled 5000 times to create the `height` and `weight` arrays, and then `column_stack` was used to paste them together as two columns.

```python
import numpy as np
height = np.round(np.random.normal(1.75, 0.2, 5000), 2)
weight = np.round(np.random.normal(60.32, 15, 5000), 2)
np_city = np.column_stack((height, weight))
```

## Matplotlib

The better you understand your data, the better you'll be able to extract insights. And once you've found those insights, again, you'll need visualization to be able to share your valuable insights with other people.

![w01_matplotlib.png](./assets/w01_matplotlib.png "w01_matplotlib.png")

There are many visualization packages in python, but the mother of them all, is `matplotlib`. You will need its subpackage `pyplot`. By convention, this subpackage is imported as `plt`:

```python
import matplotlib.pyplot as plt
```

### Line plot

Let's try to gain some insights in the evolution of the world population. To plot data as a **line chart**, we call `plt.plot` and use our two lists as arguments. The first argument corresponds to the horizontal axis, and the second one to the vertical axis.

```python
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# "plt.plot" creates the plot, but does not display it
plt.plot(year, pop)

# "plt.show" displays the plot
plt.show()
```

You'll have to call `plt.show()` explicitly because you might want to add some extra information to your plot before actually displaying it, such as titles and label customizations.

As a result we get:

![w01_matplotlib_result.png](./assets/w01_matplotlib_result.png "w01_matplotlib_result.png")

We see that:

- the years are indeed shown on the horizontal axis;
- the populations on the vertical axis;
- this type of plot is great for plotting a time scale along the x-axis and a numerical feature on the y-axis.

There are four data points, and Python draws a line between them.

![w01_matplotlib_edited.png](./assets/w01_matplotlib_edited.png "w01_matplotlib_edited.png")

In 1950, the world population was around 2.5 billion. In 2010, it was around 7 billion.

> **Insight:** The world population has almost tripled in sixty years.
>
> **Note:** If you pass only one argument to `plt.plot`, Python will know what to do and will use the index of the list to map onto the `x` axis, and the values in the list onto the `y` axis.

### Scatter plot

We can reuse the code from before and just swap `plt.plot(...)` with `plt.scatter(...)`:

```python
year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]

# "plt.plot" creates the plot, but does not display it
plt.scatter(year, pop)

# "plt.show" displays the plot
plt.show()
```

![w01_matplotlib_scatter.png](./assets/w01_matplotlib_scatter.png "w01_matplotlib_scatter.png")

The resulting scatter plot:

- plots the individual data points;
- dots aren't connected with a line;
- is great for plotting two numerical features (example: correlation analysis).

### Drawing multiple plots on one figure

This can be done by first instantiating the figure and two axis and the using each axis to plot the data. Example taken from [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots).

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.suptitle('Sharing Y axis')

ax1.plot(x, y)
ax2.scatter(x, y)

plt.show()
```

![w01_multiplot.png](./assets/w01_multiplot.png "w01_multiplot.png")

### The logarithmic scale

Sometimes the correlation analysis between two variables can be done easier when one or all of them is plotted on a logarithmic scale. This is because we would reduce the difference between large values as this scale "squashes" large numbers:

![w01_logscale.png](./assets/w01_logscale.png "w01_logscale.png")

In `matplotlib` we can use the [plt.xscale](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xscale.html) function to change the scaling of an axis using `plt` or [ax.set_xscale](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#matplotlib.axes.Axes.set_xscale) to set the scale of an axis of a subplot.

### Histogram

#### Introduction

The histogram is a plot that's useful to explore **distribution of numeric** data;

Imagine `12` values between `0` and `6`.

![w01_histogram_ex1.png](./assets/w01_histogram_ex1.png "w01_histogram_ex1.png")

To build a histogram for these values, you can divide the line into **equal chunks**, called **bins**. Suppose you go for `3` bins, that each have a width of `2`:

![w01_histogram_ex2.png](./assets/w01_histogram_ex2.png "w01_histogram_ex2.png")

Next, you count how many data points sit inside each bin. There's `4` data points in the first bin, `6` in the second bin and `2` in the third bin:

![w01_histogram_ex3.png](./assets/w01_histogram_ex3.png "w01_histogram_ex3.png")

Finally, you draw a bar for each bin. The height of the bar corresponds to the number of data points that fall in this bin. The result is a histogram, which gives us a nice overview on how the `12` values are **distributed**. Most values are in the middle, but there are more values below `2` than there are above `4`:

![w01_histogram_ex4.png](./assets/w01_histogram_ex4.png "w01_histogram_ex4.png")

#### In `matplotlib`

In `matplotlib` we can use the `.hist` function. In its documentation there're a bunch of arguments you can specify, but the first two are the most used ones:

- `x` should be a list of values you want to build a histogram for;
- `bins` is the number of bins the data should be divided into. Based on this number, `.hist` will automatically find appropriate boundaries for all bins, and calculate how may values are in each one. If you don't specify the bins argument, it will by `10` by default.

![w01_histogram_matplotlib.png](./assets/w01_histogram_matplotlib.png "w01_histogram_matplotlib.png")

The number of bins is important in the following way:

- too few bins will oversimplify reality and won't show you the details;
- too many bins will overcomplicate reality and won't show the bigger picture.

Experimenting with different numbers and/or creating multiple plots on the same canvas can alleviate that.

Here's the code that generated the above example:

```python
import matplotlib.pyplot as plt
xs = [0, 0.6, 1.4, 1.6, 2.2, 2.5, 2.6, 3.2, 3.5, 3.9, 4.2, 6]
plt.hist(xs, bins=3)
plt.show()
```

and the result of running it:

![w01_histogram_matplotlib_code.png](./assets/w01_histogram_matplotlib_code.png "w01_histogram_matplotlib_code.png")

#### Use cases

Histograms are really useful to give a bigger picture. As an example, have a look at this so-called **population pyramid**. The age distribution is shown, for both males and females, in the European Union.

![w01_population_pyramid.png](./assets/w01_population_pyramid.png "w01_population_pyramid.png")

Notice that the histograms are flipped 90 degrees; the bins are horizontal now. The bins are largest for the ages `40` to `44`, where there are `20` million males and `20` million females. They are the so called baby boomers. These are figures of the year `2010`. What do you think will have changed in `2050`?

Let's have a look.

![w01_population_pyramid_full.png](./assets/w01_population_pyramid_full.png "w01_population_pyramid_full.png")

The distribution is flatter, and the baby boom generation has gotten older. **With the blink of an eye, you can easily see how demographics will be changing over time.** That's the true power of histograms at work here!

### Checkpoint

<details>

<summary>
You want to visually assess if the grades on your exam follow a particular distribution. Which plot do you use?

```text
A. Line plot.
B. Scatter plot.
C. Histogram.
```

</summary>

Answer: C.

</details>

<details>

<summary>
You want to visually assess if longer answers on exam questions lead to higher grades. Which plot do you use?

```text
A. Line plot.
B. Scatter plot.
C. Histogram.
```

</summary>

Answer: B.

</details>

### Customization

Creating a plot is one thing. Making the correct plot, that makes the message very clear - that's the real challenge.

For each visualization, you have many options:

- change colors;
- change shapes;
- change labels;
- change axes, etc., etc.

The choice depends on:

- the data you're plotting;
- the story you want to tell with this data.

Below are outlined best practices when it comes to creating an MVP plot.

If we run the script for creating a line plot, we already get a pretty nice plot:

![w01_plot_basic.png](./assets/w01_plot_basic.png "w01_plot_basic.png")

It shows that the population explosion that's going on will have slowed down by the end of the century.

But some things can be improved:

- **axis labels**;
- **title**;
- **ticks**.

#### Axis labels

The first thing you always need to do is label your axes. We can do this by using the `xlabel` and `ylabel` functions. As inputs, we pass strings that should be placed alongside the axes.

![w01_plot_axis_labels.png](./assets/w01_plot_axis_labels.png "w01_plot_axis_labels.png")

#### Title

We're also going to add a title to our plot, with the `title` function. We pass the actual title, `'World Population Projections'`, as an argument:

![w01_plot_title.png](./assets/w01_plot_title.png "w01_plot_title.png")

#### Ticks

Using `xlabel`, `ylabel` and `title`, we can give the reader more information about the data on the plot: now they can at least tell what the plot is about.

To put the population growth in perspective, the y-axis should start from `0`. This can be achieved by using the `yticks` function. The first input is a list, in this example with the numbers `0` up to `10`, with intervals of `2`:

![w01_plot_ticks.png](./assets/w01_plot_ticks.png "w01_plot_ticks.png")

Notice how the curve shifts up. Now it's clear that already in `1950`, there were already about `2.5` billion people on this planet.

Next, to make it clear we're talking about billions, we can add a second argument to the `yticks` function, which is a list with the display names of the ticks. This list should have the same length as the first list.

![w01_plot_tick_labels.png](./assets/w01_plot_tick_labels.png "w01_plot_tick_labels.png")

#### Adding more data

Finally, let's add some more historical data to accentuate the population explosion in the last `60` years. If we run the script once more, three data points are added to the graph, giving a more complete picture.

![w01_plot_more_data.png](./assets/w01_plot_more_data.png "w01_plot_more_data.png")

#### `plt.tight_layout()`

##### Problem

With the default Axes positioning, the axes title, axis labels, or tick labels can sometimes go outside the figure area, and thus get clipped.

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.show()
```

![w01_tight_layout_1.png](./assets/w01_tight_layout_1.png "w01_tight_layout_1.png")

##### Solution

To prevent this, the location of Axes needs to be adjusted. `plt.tight_layout()` does this automatically:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ax = plt.subplots()
example_plot(ax, fontsize=24)
plt.tight_layout()
plt.show()
```

![w01_tight_layout_2.png](./assets/w01_tight_layout_2.png "w01_tight_layout_2.png")

When you have multiple subplots, often you see labels of different Axes overlapping each other:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.show()
```

![w01_tight_layout_3.png](./assets/w01_tight_layout_3.png "w01_tight_layout_3.png")

`plt.tight_layout()` will also adjust spacing between subplots to minimize the overlaps:

```python
import matplotlib.pyplot as plt
import numpy as np

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout()
plt.show()
```

![w01_tight_layout_4.png](./assets/w01_tight_layout_4.png "w01_tight_layout_4.png")

## Seaborn

Seaborn is a Python data visualization library based on `matplotlib`. It provides a higher-level interface for drawing attractive and informative statistical graphics. Most of the time what you can do in `matplotlib` with several lines of code, you can do with `seaborn` with one line! Refer to the notebook `matplotlib_vs_seaborn.ipynb` in `Week_01` to see a comparison between `matplotlib` and `seaborn`.

In the course you can use both `matplotlib` and `seaborn`! **If you want to only work with `seaborn`, this is completely fine!** Use a plotting library of your choice for any plotting exercises in the course.

## Pandas

### Introduction

As a person working in the data field, you'll often be working with tons of data. The form of this data can vary greatly, but pretty often, you can boil it down to a tabular structure, that is, in the form of a table like in a spreadsheet.

Suppose you're working in a chemical plant and have temperature measurements to analyze. This data can come in the following form:

![w01_pandas_ex1.png](./assets/w01_pandas_ex1.png "w01_pandas_ex1.png")

- row = measurement = observation;
- column = variable. There are three variables above: temperature, date and time, and the location.

- To start working on this data in Python, you'll need some kind of rectangular data structure. That's easy, we already know one! The 2D `numpy` array, right?
    > It's an option, but not necessarily the best one. There are different data types and `numpy` arrays are not great at handling these.
        - it's possible that for a single column we have both a string variable and a float variable.

Your datasets will typically comprise different data types, so we need a tool that's better suited for the job. To easily and efficiently handle this data, there's the `pandas` package:

- a high level data manipulation tool;
- developed by [Wes McKinney](https://en.wikipedia.org/wiki/Wes_McKinney);
- built on the `numpy` package, meaning it's fast;
- Compared to `numpy`, it's more high level, making it very interesting for data scientists all over the world.

In `pandas`, we store the tabular data in an object called a **DataFrame**. Have a look at this [BRICS](https://en.wikipedia.org/wiki/BRICS) data:

![w01_pandas_brics.png](./assets/w01_pandas_brics.png "w01_pandas_brics.png")

We see a similar structure:

- the rows represent the observations;
- the columns represent the variables.

Notice also that:

- each row has a unique row label: `BR` for `Brazil`, `RU` for `Russia`, and so on;
- the columns have labels: `country`, `population`, and so on;
- the values in the different columns have different types.

### DataFrame from Dictionary

One way to build dataframes is using a dictionary:

- the keys are the column labels;
- the values are the corresponding columns, in list form.

```python
import pandas as pd
data = {
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'poplulation': [200.4, 143.5, 1252, 1357, 52.98],
}

df_brics = pd.DataFrame(data)
df_brics
```

```console
        country    capital    area  poplulation
0        Brazil   Brasilia   8.516       200.40
1        Russia     Moscow  17.100       143.50
2         India  New Delhi   3.286      1252.00
3         China    Beijing   9.597      1357.00
4  South Africa   Pretoria   1.221        52.98
```

We're almost there:

- `pandas` assigned automatic row labels, `0` up to `4`;
- to specify them manually, you can set the `index` attribute of `df_brics` to a list with the correct labels.

```python
import pandas as pd
data = {
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'poplulation': [200.4, 143.5, 1252, 1357, 52.98],
}
index = ['BR', 'RU', 'IN', 'CH', 'SA']

df_brics = pd.DataFrame(data, index)
df_brics
```

or

```python
import pandas as pd
data = {
    'country': ['Brazil', 'Russia', 'India', 'China', 'South Africa'],
    'capital': ['Brasilia', 'Moscow', 'New Delhi', 'Beijing', 'Pretoria'],
    'area': [8.516, 17.10, 3.286, 9.597, 1.221],
    'poplulation': [200.4, 143.5, 1252, 1357, 52.98],
}

df_brics = pd.DataFrame(data)
df_brics.index = ['BR', 'RU', 'IN', 'CH', 'SA']
df_brics
```

```console
         country    capital    area  poplulation
BR        Brazil   Brasilia   8.516       200.40
RU        Russia     Moscow  17.100       143.50
IN         India  New Delhi   3.286      1252.00
CH         China    Beijing   9.597      1357.00
SA  South Africa   Pretoria   1.221        52.98
```

The resulting `df_brics` is the same one as we saw before.

### DataFrame from CSV file

Using a dictionary approach is fine, but what if you're working with tons of data - you won't build the DataFrame manually, right? Yep - in this case, you import data from an external file that contains all this data.

A common file format used to store such data is the `CSV` file ([comma-separated file](https://en.wikipedia.org/wiki/Comma-separated_values)).

Here are the contents of our `brics.csv` file (you can find it in the `DATA` folder):

```console
,country,capital,area,poplulation
BR,Brazil,Brasilia,8.516,200.4
RU,Russia,Moscow,17.1,143.5
IN,India,New Delhi,3.286,1252.0
CH,China,Beijing,9.597,1357.0
SA,South Africa,Pretoria,1.221,52.98
```

Let's try to import this data into Python using the `read_csv` function:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv')
df_brics
```

```console
  Unnamed: 0       country    capital    area  poplulation
0         BR        Brazil   Brasilia   8.516       200.40
1         RU        Russia     Moscow  17.100       143.50
2         IN         India  New Delhi   3.286      1252.00
3         CH         China    Beijing   9.597      1357.00
4         SA  South Africa   Pretoria   1.221        52.98
```

There's something wrong:

- the row labels are seen as a column;
- to solve this, we'll have to tell the `read_csv` function that the first column contains the row indexes. This is done by setting the `index_col` argument.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics
```

```console
         country    capital    area  poplulation
BR        Brazil   Brasilia   8.516       200.40
RU        Russia     Moscow  17.100       143.50
IN         India  New Delhi   3.286      1252.00
CH         China    Beijing   9.597      1357.00
SA  South Africa   Pretoria   1.221        52.98
```

This time `df_brics` contains the `DataFrame` we started with. The `read_csv` function features many more arguments that allow you to customize your data import, so make sure to check out its [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).

### Indexing and selecting data

#### Column Access using `[]`

Suppose you only want to select the `country` column from `df_brics`. How do we do this? We can put the column label - `country`, inside square brackets.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics['country']
```

```console
BR          Brazil
RU          Russia
IN           India
CH           China
SA    South Africa
Name: country, dtype: object
```

- we see as output the entire column, together with the row labels;
- the last line says `Name: country, dtype: object` - we're not dealing with a regular DataFrame here. Let's find out about the `type` of the object that gets returned.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
type(df_brics['country'])
```

```console
<class 'pandas.core.series.Series'>
```

- we're dealing with a `Pandas Series` here;
- you can think of the Series as a 1-dimensional NumPy array that can be labeled:
  - if you paste together a bunch of Series next to each other, you can create a DataFrame.

If you want to select the `country` column but keep the data in a DataFrame, you'll need double square brackets:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)

print(f"Type is {type(df_brics[['country']])}")

df_brics[['country']]
```

```console
Type is <class 'pandas.core.frame.DataFrame'>
         country
BR        Brazil
RU        Russia
IN         India
CH         China
SA  South Africa
```

- to select two columns, `country` and `capital`, add them to the inner list. Essentially, we're putting a list with column labels inside another set of square brackets, and end up with a 'sub DataFrame', containing only the wanted columns.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[['country', 'capital']]
```

```console
         country    capital
BR        Brazil   Brasilia
RU        Russia     Moscow
IN         India  New Delhi
CH         China    Beijing
SA  South Africa   Pretoria
```

#### Row Access using `[]`

You can also use the same square brackets to select rows from a DataFrame. The way to do it is by specifying a slice. To get the **second**, **third** and **fourth** rows of brics, we use the slice `1` colon `4` (the end of the slice is exclusive).

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[1:4]
```

```console
   country    capital    area  population
RU  Russia     Moscow  17.100       143.5
IN   India  New Delhi   3.286      1252.0
CH   China    Beijing   9.597      1357.0
```

- these square brackets work, but it only offers limited functionality:
  - we'd want something similar to 2D NumPy arrays:
    - the index or slice before the comma referred to the rows;
    - the index or slice after the comma referred to the columns.
- if we want to do a similar thing with Pandas, we have to extend our toolbox with the `loc` and `iloc` functions.

`loc` is a technique to select parts of your data **based on labels**, `iloc` is position **based**.

#### Row Access using `loc`

Let's try to get the row for Russia. We put the label of the row of interest in square brackets after `loc`.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc['RU']
```

```console
country       Russia
capital       Moscow
area            17.1
population     143.5
Name: RU, dtype: object
```

Again, we get a Pandas Series, containing all the row's information (rather inconveniently shown on different lines).

To get a DataFrame, we have to put the `'RU'` string inside another pair of brackets.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[['RU']]
```

```console
   country capital  area  population
RU  Russia  Moscow  17.1       143.5
```

We can also select multiple rows at the same time. Suppose you want to also include `India` and `China` - we can add some more row labels to the list.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[['RU', 'IN', 'CH']]
```

```console
   country    capital    area  population
RU  Russia     Moscow  17.100       143.5
IN   India  New Delhi   3.286      1252.0
CH   China    Beijing   9.597      1357.0
```

This was only selecting entire rows, that's something you could also do with the basic square brackets and integer slices.

**The difference here is that you can extend your selection with a comma and a specification of the columns of interest. This is used a lot!**

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[['RU', 'IN', 'CH'], ['country', 'capital']]
```

```console
   country    capital
RU  Russia     Moscow
IN   India  New Delhi
CH   China    Beijing
```

You can also use `loc` to select all rows but only a specific number of columns - replace the first list that specifies the row labels with a colon (this is interpreted as a slice going from beginning to end). This time, the intersection spans all rows, but only two columns.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[:, ['country', 'capital']]
```

```console
         country    capital
BR        Brazil   Brasilia
RU        Russia     Moscow
IN         India  New Delhi
CH         China    Beijing
SA  South Africa   Pretoria
```

Note that the above can be done, by simply omitting the slice which is usually done.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[['country', 'capital']]
```

```console
         country    capital
BR        Brazil   Brasilia
RU        Russia     Moscow
IN         India  New Delhi
CH         China    Beijing
SA  South Africa   Pretoria
```

#### Checkpoint

- Square brackets:
  - Column access `brics[['country', 'capital']]`;
  - Row access: only through slicing `brics[1:4]`.
- loc (label-based):
  - Row access `brics.loc[['RU', 'IN', 'CH']]`;
  - Column access `brics.loc[:, ['country', 'capital']]`;
  - Row and Column access `df_brics.loc[['RU', 'IN', 'CH'], ['country', 'capital']]` (**very powerful**).

#### `iloc`

In `loc`, you use the `'RU'` string in double square brackets, to get a DataFrame. In `iloc`, you use the index `1` instead of `RU`. The results are exactly the same.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)

print(df_brics.loc[['RU']])
print()
print(df_brics.iloc[[1]])
```

```console
   country capital  area  population
RU  Russia  Moscow  17.1       143.5

   country capital  area  population
RU  Russia  Moscow  17.1       143.5
```

To get the rows for `Russia`, `India` and `China`:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.iloc[[1, 2, 3]]
```

```console
   country    capital    area  population
RU  Russia     Moscow  17.100       143.5
IN   India  New Delhi   3.286      1252.0
CH   China    Beijing   9.597      1357.0
```

To in addition only keep the `country` and `capital` column:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.iloc[[1, 2, 3], [0, 1]]
```

```console
   country    capital
RU  Russia     Moscow
IN   India  New Delhi
CH   China    Beijing
```

To keep all rows and just select columns:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.iloc[:, [0, 1]]
```

```console
         country    capital
BR        Brazil   Brasilia
RU        Russia     Moscow
IN         India  New Delhi
CH         China    Beijing
SA  South Africa   Pretoria
```

### Filtering pandas DataFrames

#### Single condition

Suppose you now want to get the countries for which the area is greater than 8 million square kilometers. In this case, they are Brazil, Russia, and China.

```console
         country    capital    area  population
BR        Brazil   Brasilia   8.516      200.40
RU        Russia     Moscow  17.100      143.50
IN         India  New Delhi   3.286     1252.00
CH         China    Beijing   9.597     1357.00
SA  South Africa   Pretoria   1.221       52.98
```

There are three steps:

1. Get the `area` column.
2. Perform the comparison on this column and store its result.
3. Use this result to do the appropriate selection on the DataFrame.

- There are a number of ways to achieve step `1`. What's important is that we should get a Pandas Series, not a Pandas DataFrame. How can we achieve this?
    > `brics['area']` or `brics.loc[:, 'area']` or `brics.iloc[:, 2]`

Next, we actually perform the comparison. To see which rows have an area greater than `8`, we simply append `> 8` to the code:

```python
print(df_brics['area'])
print()
print(df_brics['area'] > 8)
```

```console
BR     8.516
RU    17.100
IN     3.286
CH     9.597
SA     1.221
Name: area, dtype: float64

BR     True
RU     True
IN    False
CH     True
SA    False
Name: area, dtype: bool
```

We get a Series containing booleans - areas with a value over `8` correspond to `True`, the others - to `False`.

The final step is using this boolean Series to subset the Pandas DataFrame. To do this, we put the result of the comparison inside square brackets.

```python
is_huge = df_brics['area'] > 8
df_brics[is_huge]
```

```console
   country   capital    area  population
BR  Brazil  Brasilia   8.516       200.4
RU  Russia    Moscow  17.100       143.5
CH   China   Beijing   9.597      1357.0
```

We can also write this in a one-liner. Here is the final full code:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[df_brics['area'] > 8]
```

```console
   country   capital    area  population
BR  Brazil  Brasilia   8.516       200.4
RU  Russia    Moscow  17.100       143.5
CH   China   Beijing   9.597      1357.0
```

#### Multiple conditions

Suppose you only want to keep the observations that have an area between `8` and `10` million square kilometers, inclusive. To do this, we'll have to place not one, but two conditions in the square brackets and connect them via **boolean logical operators**. The boolean logical operators we can use are:

- `&`: both conditions on each side have to evaluate to `True` (just like `&&` in `C++`);
- `|`: at least one of the conditions on each side have to evaluate to `True` (just like `||` in `C++`);
- `~`: unary operator, used for negation (just like `!` in `C++`).

And here is the Python code:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[(df_brics['area'] >= 8) & (df_brics['area'] <= 10)]
```

```console
   country   capital   area  population
BR  Brazil  Brasilia  8.516       200.4
CH   China   Beijing  9.597      1357.0
```

> **Note:** It is very important to remember that when we have multiple conditions each condition should be surrounded by round brackets `(`, `)`.

There is also quite a handy pandas function that can do this for us called [`between`](https://pandas.pydata.org/docs/reference/api/pandas.Series.between.html):

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics[df_brics['area'].between(8, 10)]
```

```console
   country   capital   area  population
BR  Brazil  Brasilia  8.516       200.4
CH   China   Beijing  9.597      1357.0
```

#### Filtering and selecting columns

You can also put these filters in `loc` and select columns - this is very handy for performing both subsetting row-wise and column-wise. This is how we can get the **countries** that have an area between `8` and `10` million square kilometers:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics.loc[(df_brics['area'] >= 8) & (df_brics['area'] <= 10), 'country']
```

```console
BR    Brazil
CH     China
Name: country, dtype: object
```

### Looping over dataframes

If a Pandas DataFrame were to function the same way as a 2D NumPy array or a list with nested lists in it, then maybe a basic for loop like the following would print out each row. Let's see what the output is.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
for val in df_brics:
    print(val)
```

```console
country
capital
area
population
```

Well, this was rather unexpected. We simply got the column names.

In Pandas, you have to mention explicitly that you want to iterate over the rows. You do this by calling the `iterrows` method - it looks at the DataFrame and on each iteration generates two pieces of data:

- the label of the row;
- the actual data in the row as a Pandas Series.

Let's change the for loop to reflect this change. We'll store:

- the row label as `lab`;
- the row data as `row`.

To understand what's happening, let's print `lab` and `row` seperately.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
for lab, row in df_brics.iterrows():
    print(f'Lab is:')
    print(lab)
    print(f'Row is:')
    print(row)
    print()
```

```console
Lab is:
BR
Row is:
country         Brazil 
capital       Brasilia 
area             8.516 
population       200.4 
Name: BR, dtype: object

Lab is:
RU
Row is:
country       Russia
capital       Moscow
area            17.1
population     143.5
Name: RU, dtype: object

Lab is:
IN
Row is:
country           India
capital       New Delhi
area              3.286
population       1252.0
Name: IN, dtype: object

Lab is:
CH
Row is:
country         China
capital       Beijing
area            9.597
population     1357.0
Name: CH, dtype: object

Lab is:
SA
Row is:
country       South Africa
capital           Pretoria
area                 1.221
population           52.98
Name: SA, dtype: object
```

Because this `row` variable on each iteration is a Series, you can easily select additional information from it using the subsetting techniques from earlier.

<details>

<summary>Suppose you only want to print out the capital on each iteration. How can this be done?
</summary>

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
for lab, row in df_brics.iterrows():
    print(f"{lab}: {row['capital']}")
```

```console
BR: Brasilia
RU: Moscow
IN: New Delhi
CH: Beijing
SA: Pretoria
```

</details>

<details>

<summary>Let's add a new column to the end, named `name_length`, containing the number of characters in the country's name. What steps could we follow to do this?
</summary>

Here's our plan of attack:

  1. The specification of the `for` loop can be the same, because we'll need both the row label and the row data;
  2. We'll calculate the length of each country name by selecting the country column from row, and then passing it to the `len()` function;
  3. We'll add this new information to a new column, `name_length`, at the appropriate location.
  4. To see whether we coded things correctly, we'll add a printout of the data after the `for` loop.

In Python we could implement the steps like this:

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
for lab, row in df_brics.iterrows():
    df_brics.loc[lab, 'name_length'] = len(row['country'])

print(df_brics)
```

</details>

```console
         country    capital    area  population  name_length
BR        Brazil   Brasilia   8.516      200.40          6.0
RU        Russia     Moscow  17.100      143.50          6.0
IN         India  New Delhi   3.286     1252.00          5.0
CH         China    Beijing   9.597     1357.00          5.0
SA  South Africa   Pretoria   1.221       52.98         12.0
```

<details>

<summary>This is all nice, but it's not especially efficient.
</summary>

We are creating a Series object on every iteration.

A way better approach if we want to calculate an entire DataFrame column by applying a function on a particular column in an element-wise fashion, is `apply()`. In this case, we don't even need a for loop.

```python
import pandas as pd
df_brics = pd.read_csv('DATA/brics.csv', index_col=0)
df_brics['name_length'] = df_brics['country'].apply(len) # alternative: df_brics['country'].apply(lambda val: len(val))
print(df_brics)
```

```console
         country    capital    area  population  name_length
BR        Brazil   Brasilia   8.516      200.40          6.0
RU        Russia     Moscow  17.100      143.50          6.0
IN         India  New Delhi   3.286     1252.00          5.0
CH         China    Beijing   9.597     1357.00          5.0
SA  South Africa   Pretoria   1.221       52.98         12.0
```

You can think of it like this:

- the `country` column is selected out of the DataFrame;
- on it `apply` calls the `len` function with each value as input;
- `apply` produces a new `array`, that is stored as a new column, `"name_length"`.

This is way more efficient, and also easier to read.

</details>

## Random numbers

### Context

Imagine the following:

- you're walking up the empire state building and you're playing a game with a friend.
- You throw a die `100` times:
  - If it's `1` or `2` you'll go one step down.
  - If it's `3`, `4`, or `5`, you'll go one step up.
  - If you throw a `6`, you'll throw the die again and will walk up the resulting number of steps.
- also, you admit that you're a bit clumsy and have a chance of `0.1%` of falling down the stairs when you make a move. Falling down means that you have to start again from step `0`.

With all of this in mind, you bet with your friend that you'll reach `60` steps high. What is the chance that you will win this bet?

- one way to solve it would be to calculate the chance analytically using equations;
- another possible approach, is to simulate this process thousands of times, and see in what fraction of the simulations that you will reach `60` steps.

We're going to opt for the second approach.

### Random generators

We have to simulate the die. To do this, we can use random generators.

```python
import numpy as np
np.random.rand() # Pseudo-random numbers
```

```console
0.026360555982748446
```

We get a random number between `0` and `1`. This number is so-called pseudo-random. Those are random numbers that are generated using a mathematical formula, starting from a **random seed**.

This seed was chosen by Python when we called the `rand` function, but you can also set this manually. Suppose we set it to `123` and then call the `rand` function twice.

```python
import numpy as np
np.random.seed(123)
print(np.random.rand())
print(np.random.rand())
```

```console
0.6964691855978616
0.28613933495037946
```

> **Note:** Set the seed in the global scope of the Python module (not in a function).

We get two random numbers, however, if call `rand` twice more ***from a new python session***, we get the exact same random numbers!

```python
import numpy as np
np.random.seed(123)
print(np.random.rand())
print(np.random.rand())
```

```console
0.6964691855978616
0.28613933495037946
```

This is funky: you're generating random numbers, but for the same seed, you're generating the same random numbers. That's why it's called pseudo-random; **it's random but consistent between runs**; this is very useful, because this ensures ***"reproducibility"***. Other people can reproduce your analysis.

Suppose we want to simulate a coin toss.

- we set the seed;
- we use the `np.random.randint()` function: it will randomly generate either `0` or `1`. We'll pass two arguments to determine the range of the generated numbers - `0` and `2` (non-inclusive on the right side).

```python
import numpy as np
np.random.seed(123)
print(np.random.randint(0, 2))
print(np.random.randint(0, 2))
print(np.random.randint(0, 2))
```

```console
0
1
0
```

We can extend the code with an `if-else` statement to improve user experience:

```python
import numpy as np
np.random.seed(123)
coin = np.random.randint(0, 2)
print(coin)
if coin == 0:
    print('heads')
else:
    print('tails')
```

```console
heads
```
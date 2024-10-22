import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

df = pd.read_csv('advertising_and_sales_clean.csv')
y = df['sales']
X = df.drop(columns=['sales', 'influencer'])
cross_val_res = cross_val_score(LinearRegression(), X, y, cv=KFold(
    n_splits=6, random_state=5, shuffle=True))

print('Mean:', cross_val_res.mean())
print('Standard Deviation:', cross_val_res.std())
print('95% Confidence Interval:', np.quantile(cross_val_res, [0.025, 0.975]))

plt.plot(list(range(1, 7)), cross_val_res)
plt.suptitle('R^2 per 6-fold split')
plt.xlabel('# Split')
plt.ylabel('R^2')
plt.show()

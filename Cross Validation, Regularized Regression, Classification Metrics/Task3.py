import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

df = pd.read_csv('advertising_and_sales_clean.csv')
y = df['sales']
X = df.drop(columns=['sales', 'influencer'])

coeficients = Lasso(0.1).fit(X, y).coef_

res = {feature: val for feature, val in zip(
    ['tv', 'radio', 'social_media'], coeficients.round(4))}
print(res)
plt.bar(['tv', 'radio', 'social_media'], coeficients.round(4))
plt.suptitle('Feature importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

print('The most important feature to predict sales is tv, because on the plot we see that the lasso regression gives more importance to the tv feature')

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('advertising_and_sales_clean.csv')
y = df['sales']
X = df.drop(columns=['sales', 'influencer'])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

alpha_values = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
results = {}
for alpha in alpha_values:
    model = Ridge(alpha)
    model.fit(X_train, y_train)
    results[alpha] = model.score(X_test, y_test)

print(results)

plt.plot(results.keys(), results.values())
plt.suptitle('R^2 per alpha')
plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.ylim(0.99, 1)
plt.show()
print('Няма нито overfitting, нито underfitting. Няма underfitting заради високите стойности на R^2, и няма overfitting защото при различни стойности на алфа R^2 не се променя')

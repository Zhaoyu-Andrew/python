# Linear Regression 
### import library
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
### quick view of the data
view count non-null values for columns
```python
USAhousing.info()
USAhousing.describe()
```
### training linear regression model
**split data into `x` and `y`**
x = data to train 
y = target variable 
**train test split data**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```
randomly select 40% of data as testing data
**creating and training the model**
```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
```
**model evaluation**
each coefficient stands for how much y will increase as each paramer increases by 1 

```python
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
```
**prediction**
if the scatter plot is a nice linear fit, the model is effective 
```python
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
```
**evaluate prediction result using evaluation metrics**
```python
from sklearn import metrics 
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```
#Logistic regression
## import library
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
## load data
```python
train = pd.read_csv('titanic_train.csv')
```
## explore data
use heatmap to check null values in the dataset
```python
# explore null values in the dataset
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# explore survived people
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
# explore survived people through sex
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
# explore survived people through people's class
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
# explore age range
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

```
## clean data
Here, different Pclass shows substantial difference for the age, so we fill the "null" age with average ages for people in different class
```python
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
```
```python
def impute_age(cols):
    Age = cols[0]

    Pclass = cols[1]
    
    if pd.isnull(Age):
        # here 37 is the average age of people in Pclass 1
        # if age is null and the person is in Pclass 1, we 
        # fill the empty age with 37.
        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
```
select both the `age` and `Pclass` columns and apply the function
```python
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
```
check null value using heatmap again
```python
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```
drop columns that will for sure have no effect on the value we want to predict 

```python
train.drop('Cabin',axis=1,inplace=True)
```
drop other `na` values in other columns
```python
train.dropna(inplace=True)
```
## convert categorical features
convert categorical data to numerical data so that we can apply the algorithm on the numerical data

```python
# check out data type of each column, type with 'object' is categorical
train.info()
```
get dummy variable, such as using '1' to represent male for important columns
```python
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
```
drop all useless columns includes ones we just apply the dummy
important columns
```python
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
```
combine original columns with dummy column
```python
train = pd.concat([train,sex,embark],axis=1)
```
## build logistic regression model
**train test split**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)
```
**training and predicting**
```python
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
```
**evaluation**
```python
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
```
**understanding classfication report**
- **precision**
Precision = percentage of true positive
Precision = TP / (TP + FP)
- **Recall**
Recall = percentage of positive cases
Precision = TP / (TP + FN)
- **F1 score**
F1 score = weighted score based on both precision and recall,ranging from 0 to 1
F1 Score = 2*(Recall * Precision) / (Recall + Precision)
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

df = pd.read_csv("./titanic_data_updated.csv")

df.drop(['PassengerID', 'Name','Ticket'],axis=1, inplace=True)

df.isnull().sum()

X = df.drop(['Survived'],axis=1)
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

display(X_train)
display(y_train)

# Numerical missing value imputation with pandas

X_train.isnull().sum()
X_test.isnull().sum()


age_mean = X_train['Age'].mean()

X_train['age_mean_imputor'] = X_train['Age'].fillna(age_mean)
X_test['age_mean_imputor'] = X_test['Age'].fillna(age_mean)

# median impute 

age_median = X_train['age_median_imputor'].fillna(age_median)

X_train['age_median_imputor'] = X_train['Age'].fillna(age_median)

X_test['age_median_imputor'] = X_test['Age'].fillna(age_median)

sns.kdeplot(data=X_train, x = 'age_mean_imputor')

sns.kdeplot(data=X_train, x='age_median_imputor')


from sklearn.impute import SimpleImputer

age_imputor = SimpleImputer(missing_values=np.nan, strategy='mean')
age_imputor.fit(X_train[['Age']])

X_train['Age'] = age_imputor.transform(X_train[['Age']]).ravel()


X_train.isnull().sum()

X_train.drop(['age_mean_imputor', 'age_median_imputor'], axis= 1, inplace = True)

X_test['Age'] = age_imputor.transform(X_test[['Age']])
X_test.isnull().sum()

X_test_drop(['age_mean_imputor', 'age_median_imputor'], axis = 1, inplace = True)



embarked_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

embarked_imputer.fit(X_train['Embarked'])


X_train['Embarked'] = embarked_imputer.transform(X_train[['Embarked']]).ravel()
X_test['Embarked'] = embarked_imputer.transform(X_train[['Embarked']]).ravel()


X_train.isnull().sum()
X_test.isnull().sum()


sns.countplot(data= X_train, x=X_train['Cabin'])



cabin_imputer = SimpleImputer(missing_values=np.nan , strategy='constant',fill_value='Missing',add_indicator=True)

cabin_imputer.fit(X_train[['Cabin']])

# transform both train and test data

X_train[['Cabin','cabin_missing_indicator']]= cabin_imputer.transform(X_train[['Cabin']])
X_test[['Cabin','cabin_missing_indicator']] = cabin_imputer.transform(X_test[['Cabin']])

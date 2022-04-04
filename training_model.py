#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('titanic/train.csv')

def get_title(name):
  if '.' in name:
    return name.split(',')[1].split('.')[0].strip()
  else:
    return 'Unknown'
titles = sorted(set([x for x in data.Name.map(lambda x: get_title(x))]))

def replace_titles(x):
  title = x['Title']
  if title in ['Capt','Col','Major']:
    return 'Officer'
  elif title in ['Jonkheer','Don','the Countess','Dona','Lady','Sir']:
    return 'Officer'
  elif title in ['the Countess','Mme','Lady']:
    return 'Mrs'
  elif title in ['Mlle','Ms']:
    return 'Miss'
  else:
    return title

data['Title'] = data['Name'].map(lambda x: get_title(x))
data['Title'] = data.apply(replace_titles, axis=1)

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna('S',inplace=True)
data.drop('Cabin',axis=1,inplace=True)
data.drop('Ticket',axis=1,inplace=True)
data.drop('Name',axis=1,inplace=True)
data.Sex.replace(('male','female'),(0,1), inplace=True)
data.Embarked.replace(('S','C','Q'),(0,1,2),inplace=True)
data.Title.replace(('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'),(0,1,2,3,4,5,6,7),inplace=True)

x = data.drop(['Survived','PassengerId'],axis=1)
y = data['Survived']
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.7)

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
filename = 'titanic_model.sav'
pickle.dump(randomforest, open(filename,'wb'))


# In[ ]:





# In[ ]:





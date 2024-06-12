import warnings 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import streamlit as st

plt.style.use('fivethirtyeight') 

# Ignore warnings
warnings.filterwarnings('ignore') 

# Load data
train = pd.read_csv('train.csv') 
test = pd.read_csv('test.csv') 

# Streamlit Title
st.title('Titanic Survival Prediction')

# Display data overview
st.header('Data Overview')
st.write("Train Data Shape:", train.shape)
st.write("Train Data Info:")
buffer = pd.DataFrame({'Column': train.columns, 'Non-null Count': train.notnull().sum(), 'Dtype': train.dtypes})
st.write(buffer)
st.write("Missing Values in Train Data:", train.isnull().sum())

# Visualization - Survivors and the dead
st.header('Survivors and the Dead')
fig, ax = plt.subplots(1, 2, figsize=(12, 4)) 
train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=False) 
ax[0].set_title('Survivors (1) and the dead (0)') 
ax[0].set_ylabel('') 
sns.countplot(x='Survived', data=train, ax=ax[1]) 
ax[1].set_ylabel('Quantity') 
ax[1].set_title('Survivors (1) and the dead (0)') 
st.pyplot(fig)

# Visualization - Survivors by sex
st.header('Survivors by Sex')
fig, ax = plt.subplots(1, 2, figsize=(12, 4)) 
train[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0]) 
ax[0].set_title('Survivors by sex') 
sns.countplot(x='Sex', hue='Survived', data=train, ax=ax[1]) 
ax[1].set_ylabel('Quantity') 
ax[1].set_title('Survived (1) and deceased (0): men and women') 
st.pyplot(fig)

# Data Cleaning and Feature Engineering
train["CabinBool"] = (train["Cabin"].notnull().astype('int')) 
test["CabinBool"] = (test["Cabin"].notnull().astype('int')) 

train = train.drop(['Cabin', 'Ticket'], axis=1) 
test = test.drop(['Cabin', 'Ticket'], axis=1) 

train = train.fillna({"Embarked": "S"}) 

train["Age"] = train["Age"].fillna(-0.5) 
test["Age"] = test["Age"].fillna(-0.5) 
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] 
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior'] 
train['AgeGroup'] = pd.cut(train["Age"], bins, labels=labels) 
test['AgeGroup'] = pd.cut(test["Age"], bins, labels=labels) 

combine = [train, test] 
for dataset in combine: 
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) 

for dataset in combine: 
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare') 
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal') 
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss') 
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss') 
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs') 

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6} 
for dataset in combine: 
    dataset['Title'] = dataset['Title'].map(title_mapping) 
    dataset['Title'] = dataset['Title'].fillna(0) 

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"} 

for x in range(len(train["AgeGroup"])): 
    if train["AgeGroup"][x] == "Unknown": 
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]] 

for x in range(len(test["AgeGroup"])): 
    if test["AgeGroup"][x] == "Unknown": 
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]] 

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7} 
train['AgeGroup'] = train['AgeGroup'].map(age_mapping) 
test['AgeGroup'] = test['AgeGroup'].map(age_mapping) 

train = train.drop(['Age'], axis=1) 
test = test.drop(['Age'], axis=1) 

sex_mapping = {"male": 0, "female": 1} 
train['Sex'] = train['Sex'].map(sex_mapping) 
test['Sex'] = test['Sex'].map(sex_mapping) 

embarked_mapping = {"S": 1, "C": 2, "Q": 3} 
train['Embarked'] = train['Embarked'].map(embarked_mapping) 
test['Embarked'] = test['Embarked'].map(embarked_mapping) 

for x in range(len(test["Fare"])): 
    if pd.isnull(test["Fare"][x]): 
        pclass = test["Pclass"][x] 
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4) 

train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4]) 
test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4]) 

train = train.drop(['Fare'], axis=1) 
test = test.drop(['Fare'], axis=1) 

# Ensure all columns are numeric
st.write("Column types after transformations:")
st.write(train.dtypes)

# Model training and prediction
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score 

predictors = train.drop(['Survived', 'PassengerId', 'Name'], axis=1) 
target = train["Survived"] 
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size=0.2, random_state=0) 

randomforest = RandomForestClassifier() 
randomforest.fit(x_train, y_train) 
y_pred = randomforest.predict(x_val) 

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2) 
st.write("Accuracy of Random Forest model:", acc_randomforest, "%")

ids = test['PassengerId'] 
predictions = randomforest.predict(test.drop(['PassengerId', 'Name'], axis=1)) 

output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions}) 

# Save the result to CSV and provide download link
output_csv = output.to_csv(index=False)
st.write("Prediction results:")
st.write(output.head())
st.download_button(label="Download CSV", data=output_csv, file_name='resultfile.csv', mime='text/csv')

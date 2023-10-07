import pandas as pd #to process csv files & make dataframes
import numpy as np #for making arrays
from sklearn.model_selection import train_test_split    #pip install -U scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#load dataset to dataframe
credit_card_data = pd.read_csv('creditcard.csv')
credit_card_data.head()

#credit_card_data.info()

#check number of missing values in each column 
sum = credit_card_data.isnull().sum()
#print(sum)

#distribution of legit transactions and fraudulent transactions
count = credit_card_data['Class'].value_counts()
# print(count)
#very unbalanced dataset -> cannot use it to feed ML model because model cannot recognise fraudulent transactions due to very less fraudulent data


#separating data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# print(legit.shape) #(row,column)
# print(fraud.shape)

#statistical measures 
ldes = legit.Amount.describe()
fdes = fraud.Amount.describe()
# print(ldes)
# print(fdes)

#compare values for both transactions
mean = credit_card_data.groupby('Class').mean()
# print(mean)

# under-sampling: build a sample dataset containing similar distribution of normal transactions and fraudulent transactions
legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample,fraud], axis=0)
# print(new_dataset.head())

n = new_dataset['Class'].value_counts()
# print(n)

new_mean = new_dataset.groupby('Class').mean() # good sample occurs when the mean before and after sample isnt too different
# print(new_mean)

#splitting the data into features and types
X = new_dataset.drop(columns = 'Class', axis=1)
Y = new_dataset['Class']

# split data into training data & testing data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=2) #stratify -> evenly split class 0/1 in training and test dataset #random state -> generator used to shuffle data
# print(X.shape, X_train.shape, X_test.shape)

#Model training - Logistic regression
model = LogisticRegression()

#training model
model.fit(X_train,Y_train)

#model evaluation - accuracy score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(training_data_accuracy)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(test_data_accuracy)

#if scores between training and test data is very different, the model might be underfitted or overfitted

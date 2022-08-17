# Required Python Machine learning Packages
import pandas as pd
import numpy as np

# For preprocessing the data

from sklearn.preprocessing import Imputer
from sklearn import preprocessing


# To split the dataset into train and test datasets

from sklearn.model_selection  import train_test_split



# To model the Gaussian Navie Bayes classifier

from sklearn.naive_bayes import GaussianNB



# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score



# to import data from csv file

triage_df = pd.read_csv('dataset2.csv', header = None, delimiter=' *, *', engine='python' )


#for adding headers to csv file

triage_df.columns = ['Age', 'Gender', 'Pulse', 'B/P', 'Temperature', 'Alert', 'Voice Responsive', 'Pain Responsive','Unconscious', 'AirwayBreathing', 'Oxysat','Triage_level']

triage_df_rev = pd.DataFrame()

triage_df_rev = triage_df.copy()

h = triage_df_rev.head()
print(h)


print("Gender' : ",triage_df_rev['Gender'].unique())
print("Alert' : ",triage_df_rev['Alert'].unique())
print("Unconscious' : ",triage_df_rev['Unconscious'].unique())
print("AirwayBreathing' : ",triage_df_rev['AirwayBreathing'].unique())
print("Triage_level' : ",triage_df_rev['Triage_level'].unique())


le = preprocessing.LabelEncoder()
triage_df_rev['Gender'] = le.fit_transform(triage_df_rev['Gender'])
triage_df_rev['Alert'] = le.fit_transform(triage_df_rev['Alert'])
triage_df_rev['Unconscious'] = le.fit_transform(triage_df_rev['Unconscious'])
triage_df_rev['AirwayBreathing'] = le.fit_transform(triage_df_rev['AirwayBreathing'])
triage_df_rev['Triage_level'] = le.fit_transform(triage_df_rev['Triage_level'])
h1 = triage_df_rev.head()
print(h1)


#standardization
num_features = ['Age', 'Gender', 'Pulse', 'B/P', 'Temperature', 'Alert', 'Voice Responsive', 'Pain Responsive','Unconscious', 'AirwayBreathing', 'Oxysat','Triage_level']

scaled_features = {}
for each in num_features:
    mean, std = triage_df_rev[each].mean(),triage_df_rev[each].std()
    scaled_features[each] = [mean, std]
    triage_df_rev.loc[:, each] = (triage_df_rev[each] - mean) / std


#split this data into labels and features
target=triage_df_rev.Triage_level
data=triage_df_rev.drop('Triage_level',axis=1)

data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)
m =data_train.head()
print(m)
s=data_train.shape
print("training data",s)

s2=data_test.shape
print("testing data",s2)



#GaussianNB implementation
#create an object of the type GaussianNB
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
print("trige level predicted",pred.tolist())

#Accuracy of our Gaussian Naive Bayes model

#final_prediction=accuracy_score(target_test, pred, normalize = True)
#print("accuracy_score",final_prediction)

# Save model
from sklearn.externals import joblib
joblib.dump(gnb, 'model.pkl')
print("Model dumped!")

# Load the model that just saved
pred = joblib.load('model.pkl')


# Saving the data columns from training
model_columns = list(data.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
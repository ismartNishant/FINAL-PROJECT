"""# Required Python Machine learning Packages
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

data_fr = pd.read_csv('test.csv', header = None, delimiter=' *, *', engine='python')

data_fr.columns = ['Age', 'Gender', 'Pulse', 'B/P', 'Temperature','Alert', 'Voice Responsive', 'Pain Responsive','Unconscious', 'AirwayBreathing', 'Oxysat']

show = data_fr.isnull().sum()
print(show)

show1 = data_fr.describe(include= 'all')
print(show1)

# Organize our data
label_names = triage_df_rev.loc['target_names']
labels = triage_df_rev.loc['target']
feature_names = triage_df_rev.loc['feature_names']
features = triage_df_rev.loc['data']

print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])

#handeling missing data

show = triage_df.isnull().sum()
print(show)


#for value in ['Age', 'Gender', 'Pulse', 'B/P', 'Temperature','Alert', 'Voice Responsive', 'Pain Responsive', 'Unconscious', 'AirwayBreathing', 'Oxysat']:
#    print(value,":", sum(triage_df[value] == '?'))

#data preprocessing
triage_df_rev = pd.DataFrame()

triage_df_rev = triage_df.copy()

h = triage_df_rev.head()
print(h)

print("Gender' : ",triage_df_rev['Gender'].unique())
print("Alert' : ",triage_df_rev['Alert'].unique())
print("Unconscious' : ",triage_df_rev['Unconscious'].unique())
print("AirwayBreathing' : ",triage_df_rev['AirwayBreathing'].unique())



le = preprocessing.LabelEncoder()
triage_df_rev['Gender'] = le.fit_transform(triage_df_rev['Gender'])
triage_df_rev['Alert'] = le.fit_transform(triage_df_rev['Alert'])
triage_df_rev['Unconscious'] = le.fit_transform(triage_df_rev['Unconscious'])
triage_df_rev['AirwayBreathing'] = le.fit_transform(triage_df_rev['AirwayBreathing'])
h1 = triage_df_rev.head()
print(h1)


#standardization
num_features = ['Age', 'Gender', 'Pulse', 'B/P', 'Temperature','Alert', 'Voice Responsive', 'Pain Responsive','Unconscious', 'AirwayBreathing', 'Oxysat',]

scaled_features = {}
for each in num_features:
    mean, std = triage_df_rev[each].mean(), triage_df_rev[each].std()
    scaled_features[each] = [mean, std]
    triage_df_rev.loc[:, each] = (triage_df_rev[each] - mean) / std



#spliting


features = triage_df_rev.iloc[:,0:5].values
target = triage_df_rev.iloc[:,:5].values
features_train, features_test, target_train, target_test = train_test_split(features,target, test_size=0.2, random_state=42)

#GaussianNB implementation
#create an object of the type GaussianNB
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(features_train, target_train).predict(features_test)
#print(pred.tolist())

#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))



#summary statistics of all atributes
#show1 = triage_df_rev.describe(include= 'all')
#print(show1)


#data imputation setup
for value in ['Age', 'Gender', 'Pulse', 'B/P', 'Temperature','Alert', 'Voice Responsive', 'Pain Responsive',
                    'Unconscious', 'AirwayBreathing', 'Oxysat']:
    triage_df_rev[value].replace(['?'], [triage_df_rev.describe(include='all')[value][2]], inplace=True)

le = preprocessing.LabelEncoder()
Gender_cat = le.fit_transform(triage_df.Gender)
Alert_cat = le.fit_transform(triage_df.Alert)
Unconscious_cat = le.fit_transform(triage_df.Unconscious)
AirwayBreathing_cat = le.fit_transform(triage_df.AirwayBreathing)

#initialize the encoded categorical columns
triage_df_rev['Gender_cat'] = Gender_cat
triage_df_rev['Alert_cat'] = Alert_cat
triage_df_rev['Unconscious_cat'] = Unconscious_cat
triage_df_rev['AirwayBreathing_cat'] = AirwayBreathing_cat


#drop the old categorical columns from dataframe
dummy_fields = ['Gender_cat', 'Alert_cat', 'Unconscious_cat',  'AirwayBreathing_cat']
triage_df_rev = triage_df_rev.drop(dummy_fields, axis = 1)

show2 = triage_df_rev.head()
#print("show2")
#print(show2)


#reindexing columns


triage_df_rev = triage_df_rev.reindex(['Age', 'Gender_cat', 'Pulse', 'B/P', 'Temperature','Alert_cat', 'Voice Responsive', 'Pain Responsive','Unconscious_cat', 'AirwayBreathing_cat', 'Oxysat'],axis="columns" )

show_data = triage_df_rev.head(1)
print(show_data)



# to standardise data

num_features = ['Age', 'Gender_cat', 'Pulse', 'B/P', 'Temperature','Alert_cat', 'Voice Responsive', 'Pain Responsive','Unconscious_cat', 'AirwayBreathing_cat', 'Oxysat']

scaled_features = {}
for each in num_features:
    mean, std = triage_df_rev[each].mean(), triage_df_rev[each].std()
    scaled_features[each] = [mean, std]
    triage_df_rev.loc[:, each] = (triage_df_rev[each] - mean)/std

  #  print(triage_df_rev.loc[:, each])

 # to split data into training  set and test set

features = triage_df_rev.values[:, :11]
target = triage_df_rev.values[:, :11]
features_train, features_test, target_train, target_test = train_test_split(features,target, test_size=0.33, random_state=42)

#X_test.fillna(X_test.mean())
#np.where(x.values >= np.finfo(np.float64).max)

#Gaussian Naive Bayes Implementation

clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)


#Accuracy of our Gaussian Naive Bayes model

final_prediction=accuracy_score(target_test, target_pred, normalize = True)

features = triage_df_rev.iloc[:,0:5].values
target = triage_df_rev.iloc[:,:5].values
features_train, features_test, target_train, target_test = train_test_split(features,target, test_size=0.2, random_state=42)

sourcevars = "dataset[:,:-1] #all columns except the last one
targetvar = dataset[:,len(dataset[0])-1]"""
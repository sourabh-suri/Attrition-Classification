import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import csv
from openpyxl import load_workbook
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from openpyxl import load_workbook
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import svm, tree, linear_model, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import os


# Reading Source Files and getting their paths.
sourcefile = pd.read_csv('train.csv')
sourcefile_test = pd.read_csv('test.csv')

# Make a copy of the original sourcefile
Train_Data = sourcefile.copy()
Test_Data = sourcefile_test.copy()
ID = Train_Data['ID'].copy()
ID_test = []
ID_test=Test_Data['ID'].copy()

# Encoding
    # Training Data
Encode_obj = LabelEncoder()
for col in Train_Data.columns[1:]:
    if Train_Data[col].dtype == 'object': # Label Encoding for atmost two levels
        if len(list(Train_Data[col].unique())) <= 2: 
            Encode_obj.fit(Train_Data[col])
            Train_Data[col] = Encode_obj.transform(Train_Data[col])
    # Test Data
Encode_test = LabelEncoder()
for col in Test_Data.columns[1:]:
    if Test_Data[col].dtype == 'object':
        if len(list(Test_Data[col].unique())) <= 2:
            Encode_test.fit(Test_Data[col])
            Test_Data[col] = Encode_test.transform(Test_Data[col])
    # Converting rest of categorical variable to dummy
Train_Data = pd.get_dummies(Train_Data, drop_first=True)
Test_Data = pd.get_dummies(Test_Data, drop_first=True)
print(Train_Data.shape)

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
    # Train
Data_col = list(Train_Data.columns)
Data_col.remove('Attrition')
for col in Data_col:
    Train_Data[col] = Train_Data[col].astype(float)
    Train_Data[[col]] = scaler.fit_transform(Train_Data[[col]])
Train_Data['Attrition'] = pd.to_numeric(Train_Data['Attrition'], downcast='float')
    # Test
Test_Col = list(Test_Data.columns)
for col in Test_Col:
    Test_Data[col] = Test_Data[col].astype(float)
    Test_Data[[col]] = scaler.fit_transform(Test_Data[[col]])

# Copying Target and ID
target = Train_Data['Attrition'].copy()

# Removing Target and redundant features

Train_Data.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber','ID','MonthlyRate','HourlyRate'], axis=1, inplace=True)
Test_Data.drop(['EmployeeCount', 'EmployeeNumber','ID','MonthlyRate','HourlyRate'], axis=1, inplace=True)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,max_iter=500, multi_class='auto', n_jobs=None, penalty='l2', random_state=None, solver='lbfgs', tol=0.0001, verbose=0)
print("Logistic Regression : ",cross_val_score(logmodel, Train_Data, target, cv=5, scoring='accuracy').mean())
logmodel.fit(Train_Data,target)
predictions_LR = logmodel.predict(Test_Data)

#KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='brute',leaf_size=30, p=2, metric='minkowski')                       
print("KNeighborsClassifier : ",cross_val_score(knn, Train_Data, target, cv=20, scoring='accuracy').mean())
knn.fit(Train_Data,target)
predictions_KNN = knn.predict(Test_Data)

#Support Vector Machine
SVM=SVC(C=1.0, kernel='linear', degree=1, gamma='auto', 
        coef0=0.0, shrinking=True, probability=False, 
        tol=0.001, cache_size=200, class_weight=None, 
        verbose=False, max_iter=-1, decision_function_shape='ovo')
#SVM=svm.SVC(kernel='linear', C=1)
print("Support Vector Machine : ",cross_val_score(SVM, Train_Data, target, cv=150, scoring='accuracy').mean())
SVM.fit(Train_Data,target)
predictions_SVM = SVM.predict(Test_Data)

# Random Forest
RF = RandomForestClassifier(max_depth=50, random_state=0)
print("Random Forest : ",cross_val_score(RF, Train_Data, target, cv=10, scoring='accuracy').mean())
RF.fit(Train_Data,target)
predictions_RF = RF.predict(Test_Data)

# Decision Tree Classifier
DTC = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=50, min_samples_split=2, 
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
                             random_state=None)
print("Decision Tree Classifier : ",cross_val_score(DTC, Train_Data, target, cv=10, scoring='accuracy').mean())
DTC.fit(Train_Data,target)
predictions_DTC = DTC.predict(Test_Data)

# Neural Network- Multi-layer Perceptron Classifier
MLP = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, 
                    batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                    max_iter=2500, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False)

print("Multi-layer Perceptron Classifier : ",cross_val_score(MLP, Train_Data, target, cv=50, scoring='accuracy').mean())
MLP.fit(Train_Data,target)
predictions_MLP = MLP.predict(Test_Data)


# Gaussian Naive Bayes 
NB = GaussianNB()
print("Gaussian Naive Bayes : ",cross_val_score(NB, Train_Data, target, cv=10, scoring='accuracy').mean())
NB.fit(Train_Data,target)
predictions_NB = NB.predict(Test_Data)






#Output Generation

# Take the prediction value based on Classifier...
#predictions=np.asarray(predictions_LR)
#predictions=np.asarray(predictions_KNN)
predictions=np.asarray(predictions_SVM)
#predictions=np.asarray(predictions_RF)
#predictions=np.asarray(predictions_DTC)
#predictions=np.asarray(predictions_MLP)
#predictions=np.asarray(predictions_NB)



# Generating Output Vector
output=[]
output=np.asarray(output)
ID_test=np.asarray(ID_test)


temp= np.column_stack((ID_test, predictions))
output = np.append(output,temp)
output = np.reshape(output,(ID_test.shape[0],2))

# Write to CSV file
row_list = ['ID', 'Attrition']
with open('A_2_Output.csv', 'w') as file:
    writer = csv.writer(file,delimiter=',')
    writer.writerow(row_list)
    writer.writerows(output)

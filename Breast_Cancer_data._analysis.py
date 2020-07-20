#!/usr/bin/env python
# coding: utf-8

# In[82]:


#IMPORT ALL THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score




# In[83]:


#PRE DATA PROCESSING

#read data
columns = [ "id_number","Clump_Thickness","Uniformity_of_Cell_Size","Uniformity_of_Cell_Shape","Marginal_Adhesion",
"Single_Epithelial_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitose","Class"]

df = pd.read_csv('C:\\Users\\ASUS\\Downloads\\breast-cancer-wisconsin.data', sep=",",names=columns)
#df.to_csv (r'C:\\Users\\ASUS\\Downloads\\Breast-Cancer-Wisconsin.csv', index=None

df.head()


# In[84]:


df.info()


# In[85]:


df['Bare_Nuclei'].value_counts()


# In[86]:


df.replace('?',1,inplace=True)
#drop the id column as it does not effect the output
df.drop(['id_number'],1,inplace=True)
#change the data type 
df['Bare_Nuclei'] = df['Bare_Nuclei'].astype(np.int64)


# In[ ]:





# In[87]:


df['Class'].value_counts()


# In[88]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[89]:


df.info()


# In[90]:


df.shape


# In[91]:


df.isnull()


# In[92]:


df.isnull().sum().sum()


# In[93]:


features_mean= list(df.columns[0:8])
features_mean


# In[94]:


# Box plot explaining how the malignant or benign tumors cells can have (or not) different values for 
#the features plotting the distribution of each type of diagnosis for each of the mean features.
plt.figure(figsize=(10,10))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    
    plt.subplot(rows, 2, i+1)
    
    sns.boxplot(x='Class', y=feature, data=df, palette="Set1")

plt.tight_layout()
plt.show()


# In[95]:


X=np.array(df.drop(['Class'],1)) #independent variables (input)
y=np.array(df['Class']) #output variables


# In[96]:


X,y


# In[97]:


# Splitting the dataset into the Training set and Test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[98]:


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[99]:


X_train


# In[116]:



#MODEL FITING

Accuracy_CV=[]
Accuracy_WO_CV=[]
model_names=[]
Precisions=[]
Weighted_Precision=[]
Recall=[]
trainning_accuracy=[]

#FIND NUMBER OF BEST PARAMETERS & BEST ESTIMATOR USING CROSS VALIDATION FOR KNN
max_class=20
grid_param = {'n_neighbors': range(1, max_class)}
model = KNeighborsClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2)
clf = GridSearchCV(model, grid_param, cv=cv, scoring='accuracy')
clf.fit(X, y)
print("Best Estimator: \n{}\n".format(clf.best_estimator_))
print("Best Parameters : \n{}\n".format(clf.best_params_))
print("Best Score: \n{}\n".format(clf.best_score_))
Accuracy_CV.append(clf.best_score_)


# In[101]:


# FINDINGING THE BEST FIT K FOR KNN WITHOUT CROSS_VALIDATION
knn = []
KNN=[]
for i in range(1,21):
    classifier = KNeighborsClassifier(n_neighbors=i)
    trained_model=classifier.fit(X_train,y_train)
    trained_model.fit(X_train,y_train )
    
    # Predicting the Test set results
    
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    cm_KNN = confusion_matrix(y_test, y_pred)

    KNN.append(accuracy_score(y_test, y_pred))

    knn.append([accuracy_score(y_test, y_pred),i])

accuracy=max(KNN)
for l in knn:
    if l[0]==accuracy:
        k=l[1]
        break
Accuracy_WO_CV.append(accuracy) 
model_names.append("KNN")

print ("best of K using elbow_method:",k,",accuracy",accuracy)
print("accuracy score for tranning data:",accuracy_score(y_train, classifier.predict(X_train)))
#trainning_accuracy.append(accuracy_score(y_train, classifier.predict(X_train)))
print ("classification_report :",classification_report(y_test,y_pred))

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 21),KNN, color='red', linestyle='dashed', marker='o',  
             markerfacecolor='blue', markersize=10)
plt.title('Accuracy for different  K Value')  
plt.xlabel('K Value')  
plt.ylabel('Accuracy') 


# In[102]:


#Fitting SVC to the Training set WITHOUT CROSS VALIDATION
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

#ACCURACY WITHOUT CROSS_VALIDATION
accuracy = accuracy_score(y_test,y_pred)
print ("ACCURACY WITHOUT CROSS_VALIDATION :",accuracy)
Accuracy_WO_CV.append(accuracy)
print("accuracy score for tranning data:",accuracy_score(y_train, classifier.predict(X_train)))
trainning_accuracy.append(accuracy_score(y_train, classifier.predict(X_train)))

print ("classification_report :",classification_report(y_test,y_pred))

#CHECK ACCURACY WITH CROSS-VALIDATION(=10)
acc_cv=cross_val_score(classifier, X, y, cv=10, scoring='accuracy').mean()
print ("ACCURACY WITH CROSS_VALIDATION :",acc_cv)
Accuracy_CV.append(acc_cv) 
model_names.append("SVC")


# In[103]:


#Decision_Tree

classifier =DecisionTreeClassifier(random_state=44,max_depth=4,criterion = 'gini')
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

#ACCURACY WITHOUT CROSS_VALIDATION
accuracy = accuracy_score(y_test,y_pred)
print ("ACCURACY WITHOUT CROSS_VALIDATION :",accuracy)
Accuracy_WO_CV.append(accuracy)
print("accuracy score for tranning data:",accuracy_score(y_train, classifier.predict(X_train)))
trainning_accuracy.append(accuracy_score(y_train, classifier.predict(X_train)))
#print ("classification_report :",classification_report(y_test,y_pred))

#CHECK ACCURACY WITH CROSS-VALIDATION(=10)
acc_cv=cross_val_score(classifier, X, y, cv=10, scoring='accuracy').mean()
print ("ACCURACY WITH CROSS_VALIDATION :",acc_cv)
Accuracy_CV.append(acc_cv) 
model_names.append("DT")


# In[104]:


#Random_forest
classifier =RandomForestClassifier(n_estimators=5,criterion = 'entropy')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

#ACCURACY WITHOUT CROSS_VALIDATION
accuracy = accuracy_score(y_test,y_pred)
print ("ACCURACY WITHOUT CROSS_VALIDATION :",accuracy)
Accuracy_WO_CV.append(accuracy)
print("accuracy score for tranning data:",accuracy_score(y_train, classifier.predict(X_train)))
trainning_accuracy.append(accuracy_score(y_train, classifier.predict(X_train)))

print ("classification_report :",classification_report(y_test,y_pred))

#CHECK ACCURACY WITH CROSS-VALIDATION(=10)
acc_cv=cross_val_score(classifier, X, y, cv=10, scoring='accuracy').mean()
print ("ACCURACY WITH CROSS_VALIDATION :",acc_cv)
Accuracy_CV.append(acc_cv) 
model_names.append("RF")


# In[105]:


#Gaussian
classifier =GaussianNB()
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

#ACCURACY WITHOUT CROSS_VALIDATION
accuracy = accuracy_score(y_test,y_pred)
print ("ACCURACY WITHOUT CROSS_VALIDATION :",accuracy)
Accuracy_WO_CV.append(accuracy)
print("accuracy score for tranning data:",accuracy_score(y_train, classifier.predict(X_train)))
trainning_accuracy.append(accuracy_score(y_train, classifier.predict(X_train)))

print ("classification_report :",classification_report(y_test,y_pred))

#CHECK ACCURACY WITH CROSS-VALIDATION(=10)
acc_cv=cross_val_score(classifier, X, y, cv=10, scoring='accuracy').mean()
print ("ACCURACY WITH CROSS_VALIDATION :",acc_cv)
Accuracy_CV.append(acc_cv) 
model_names.append("NB")


# In[110]:


#.Logistic Regression
classifier =LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

#ACCURACY WITHOUT CROSS_VALIDATION
accuracy = accuracy_score(y_test,y_pred)
print ("ACCURACY WITHOUT CROSS_VALIDATION :",accuracy)
Accuracy_WO_CV.append(accuracy)
print("accuracy score for tranning data:",accuracy_score(y_train, classifier.predict(X_train)))
trainning_accuracy.append(accuracy_score(y_train, classifier.predict(X_train)))

print ("classification_report :",classification_report(y_test,y_pred))

#CHECK ACCURACY WITH CROSS-VALIDATION(=10)
acc_cv=cross_val_score(classifier, X, y, cv=10, scoring='accuracy').mean()
print ("ACCURACY WITH CROSS_VALIDATION :",acc_cv)
Accuracy_CV.append(acc_cv) 
model_names.append("LG")


# In[111]:


print(Accuracy_WO_CV)
print(Accuracy_CV)
print(model_names)


# In[112]:


ddf = pd.DataFrame(list(zip(model_names,Accuracy_WO_CV,Accuracy_CV,trainning_accuracy)), 
               columns =['models', 'Accuracy_without_cv','CV_acc','x_train_ytrain_acc']) 
ddf


# In[113]:


plt.style.use('ggplot')
n = len(ddf["Accuracy_without_cv"])
cv_acc= ddf["CV_acc"]
A = ddf["Accuracy_without_cv"]
fig, ax = plt.subplots()
index = np.arange(n)
bar_width = 0.25
opacity = 0.9
ax.bar(index, cv_acc, bar_width, alpha=opacity, color='r',
                label='cv_acc')
ax.bar(index+bar_width, A, bar_width, alpha=opacity, color='b',
                label='Accuracy_without_cv')

ax.set_xlabel('MODEL_NAMES')
ax.set_ylabel('ACCURACY')
ax.set_title('CV_ACC v/s Accuracy_without_cv')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(("KNN","SVC","DT","RF","NB","LG"))
ax.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 20:12:33 2019

@author: fahadtariq
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Importing our dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
X_train = x_sc.fit_transform(X_train)
X_test = x_sc.transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression()
classifier1.fit(X_train, y_train)
y_prob1 = classifier1.predict_proba(X_test)
y_pred1 = classifier1.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred1)

# KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5)
classifier2.fit(X_train, y_train)
y_prob2 = classifier2.predict_proba(X_test)
y_pred2 = classifier2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)

# SVM algortihm
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear', probability = True)
classifier3.fit(X_train, y_train)
y_prob3 = classifier3.predict_proba(X_test)
y_pred3 = classifier3.predict(X_test)
cm3 = confusion_matrix(y_test, y_pred3)

# SVM Kernel (Gaussian RBF Kernel) Algo : landmark optimized (Strongest so far)
from sklearn.svm import SVC
classifier4 = SVC(kernel = 'rbf', probability = True)
classifier4.fit(X_train, y_train)
y_prob4 = classifier4.predict_proba(X_test)
y_pred4 = classifier4.predict(X_test)
cm4 = confusion_matrix(y_test, y_pred4)

# Baye's theorum based algorithm
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, y_train)
y_prob5 = classifier5.predict_proba(X_test)
y_pred5 = classifier5.predict(X_test)
cm5 = confusion_matrix(y_test, y_pred5)

# rubbish overfitting
from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier6.fit(X_train, y_train)
y_prob6 = classifier6.predict_proba(X_test)
y_pred6 = classifier6.predict(X_test)
cm6 = confusion_matrix(y_test, y_pred6)

# rubbish overfitting but a lot useful
from sklearn.ensemble import RandomForestClassifier
classifier7 = RandomForestClassifier(n_estimators = 50, criterion ="entropy")
classifier7.fit(X_train, y_train)
y_prob7 = classifier7.predict_proba(X_test)
y_pred7 = classifier6.predict(X_test)
cm7 = confusion_matrix(y_test, y_pred7)

# False Positive Rate(FPR) and True Positive Rate(TPR)

# for Logistic Regression
fpr1, tpr1, threshold1 = roc_curve(y_test, y_pred1)
# Area under the curve for Logictic Regression
roc_auc1 = auc(fpr1, tpr1)

# for K-Nearest-Neighbors
fpr2, tpr2, threshold2 = roc_curve(y_test, y_pred2)
# Area under the curve for K-Nearest Neighbors
roc_auc2 = auc(fpr2, tpr2)

# for SVM
fpr3, tpr3, threshold3 = roc_curve(y_test, y_pred3)
# Area under the curve for SVM
roc_auc3 = auc(fpr3, tpr3)

# for SVM-Kernel (Gaussian RBF Kernel) --BEST SO FAR
fpr4, tpr4, threshold4 = roc_curve(y_test, y_pred4)
# Area under the curve for SVM-Kernel Algorithm
roc_auc4 = auc(fpr4, tpr4)

# for Naive-Baye's (GaussianNB)
fpr5, tpr5, threshold5 = roc_curve(y_test, y_pred5)
# Area under the curve for Naive-Baye's Algorithm
roc_auc5 = auc(fpr5, tpr5)

# for Decision Tree
fpr6, tpr6, threshold6 = roc_curve(y_test, y_pred6)
# Area under the curve for Decition Tree Algorithm
roc_auc6 = auc(fpr6, tpr6)

# for Random Forest Classifier
fpr7, tpr7, threshold7 = roc_curve(y_test, y_pred7)
# Area under the curve for Random Forest Algorithm
roc_auc7 = auc(fpr7, tpr7)

roc_auc = np.array([[roc_auc1, roc_auc2, roc_auc3, roc_auc4, roc_auc5, roc_auc6, roc_auc7]])

for i in range(100) :
    
    #Logistic Regression
    # from sklearn.linear_model import LogisticRegression
    classifier1 = LogisticRegression()
    classifier1.fit(X_train, y_train)
    y_prob1 = classifier1.predict_proba(X_test)
    y_pred1 = classifier1.predict(X_test)
    cm1 = confusion_matrix(y_test, y_pred1)
    
    # KNN Algorithm
    # from sklearn.neighbors import KNeighborsClassifier
    classifier2 = KNeighborsClassifier(n_neighbors = 5)
    classifier2.fit(X_train, y_train)
    y_prob2 = classifier2.predict_proba(X_test)
    y_pred2 = classifier2.predict(X_test)
    cm2 = confusion_matrix(y_test, y_pred2)
    
    # SVM algortihm
    # from sklearn.svm import SVC
    classifier3 = SVC(kernel = 'linear', probability = True)
    classifier3.fit(X_train, y_train)
    y_prob3 = classifier3.predict_proba(X_test)
    y_pred3 = classifier3.predict(X_test)
    cm3 = confusion_matrix(y_test, y_pred3)
    
    # SVM Kernel (Gaussian RBF Kernel) Algo : landmark optimized (Strongest so far)
    # from sklearn.svm import SVC
    classifier4 = SVC(kernel = 'rbf', probability = True)
    classifier4.fit(X_train, y_train)
    y_prob4 = classifier4.predict_proba(X_test)
    y_pred4 = classifier4.predict(X_test)
    cm4 = confusion_matrix(y_test, y_pred4)
    
    # Baye's theorum based algorithm
    # from sklearn.naive_bayes import GaussianNB
    classifier5 = GaussianNB()
    classifier5.fit(X_train, y_train)
    y_prob5 = classifier5.predict_proba(X_test)
    y_pred5 = classifier5.predict(X_test)
    cm5 = confusion_matrix(y_test, y_pred5)
    
    # rubbish overfitting
    # from sklearn.tree import DecisionTreeClassifier
    classifier6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier6.fit(X_train, y_train)
    y_prob6 = classifier6.predict_proba(X_test)
    y_pred6 = classifier6.predict(X_test)
    cm6 = confusion_matrix(y_test, y_pred6)
    
    # rubbish overfitting but a lot useful
    # from sklearn.ensemble import RandomForestClassifier
    classifier7 = RandomForestClassifier(n_estimators = 50, criterion ="entropy")
    classifier7.fit(X_train, y_train)
    y_prob7 = classifier7.predict_proba(X_test)
    y_pred7 = classifier6.predict(X_test)
    cm7 = confusion_matrix(y_test, y_pred7)
    
    # False Positive Rate(FPR) and True Positive Rate(TPR)
    
    # for Logistic Regression
    fpr1, tpr1, threshold1 = roc_curve(y_test, y_pred1)
    # Area under the curve for Logictic Regression
    roc_auc1 = auc(fpr1, tpr1)
    
    # for K-Nearest-Neighbors
    fpr2, tpr2, threshold2 = roc_curve(y_test, y_pred2)
    # Area under the curve for K-Nearest Neighbors
    roc_auc2 = auc(fpr2, tpr2)
    
    # for SVM
    fpr3, tpr3, threshold3 = roc_curve(y_test, y_pred3)
    # Area under the curve for SVM
    roc_auc3 = auc(fpr3, tpr3)
    
    # for SVM-Kernel (Gaussian RBF Kernel) --BEST SO FAR
    fpr4, tpr4, threshold4 = roc_curve(y_test, y_pred4)
    # Area under the curve for SVM-Kernel Algorithm
    roc_auc4 = auc(fpr4, tpr4)
    
    # for Naive-Baye's (GaussianNB)
    fpr5, tpr5, threshold5 = roc_curve(y_test, y_pred5)
    # Area under the curve for Naive-Baye's Algorithm
    roc_auc5 = auc(fpr5, tpr5)
    
    # for Decision Tree
    fpr6, tpr6, threshold6 = roc_curve(y_test, y_pred6)
    # Area under the curve for Decition Tree Algorithm
    roc_auc6 = auc(fpr6, tpr6)
    
    # for Random Forest Classifier
    fpr7, tpr7, threshold7 = roc_curve(y_test, y_pred7)
    # Area under the curve for Random Forest Algorithm
    roc_auc7 = auc(fpr7, tpr7)
    
    roc_auc += np.array([[roc_auc1, roc_auc2, roc_auc3, roc_auc4, roc_auc5, roc_auc6, roc_auc7]])


print(roc_auc / 111)


# ROC of each and every Classifier

# ROC for Logistic Regression
plt.plot(fpr1, tpr1, 'b', label = 'LR AUC = %0.2f' % roc_auc1)
# ROC for K-Nearest-Neighbors
plt.plot(fpr2, tpr2, 'r', label = 'KNN AUC = %0.2f' % roc_auc2)
# ROC for SVM
plt.plot(fpr3, tpr3, 'g', label = 'SVM AUC = %0.2f' % roc_auc3)
# ROC for SVM-Kernel ( Gaussian RBF Kernel)
plt.plot(fpr4, tpr4, 'y', label = 'SVMRBF AUC = %0.2f' % roc_auc4)
# ROC for Naive-Baye's (GaussianNB)
plt.plot(fpr5, tpr5, 'o', label = 'NB AUC = %0.2f' % roc_auc5)
# ROC for Decision Tree
plt.plot(fpr6, tpr6, 'm', label = 'DT AUC = %0.2f' % roc_auc6)
# ROC for Random Forest Classifier
plt.plot(fpr7, tpr7, color="black", label = 'RFC AUC = %0.2f' % roc_auc7)


# Graph
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

"""

After randonly making over a 100(*7) classifiers and averaging the values
of area under the curve for each classifier, this is the amazing result!

Results : 
    
    1.) SVM Kernel (Gaussian RBF Kernel)        0.94753505
    2.) K-Nearest Neighbors                     0.94007237
    3.) Naive Baye's                            0.88692899
    4.) Decision Tree                           0.85662596
    5.) Random Forest                           0.85662596
    6.) SVM                                     0.83378562
    7.) Logistic Regression                     0.82632293    
    
BEST CHOICE :     SVM Kernel ( Gaussian RBF Kernel)

(The values under the curves change randomly but
still the result doesn't vary too much.
After many tries, I concluded that the RBF kernel is the best
for *this dataset.)

"""

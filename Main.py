'''
Created on 31/07/2017

@author: olivia
'''
from sklearn.metrics.classification import accuracy_score
'''
Created on 31/07/2017

@author: olivia
'''
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


#[height, weight, shoe size]
X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],
     [166,65,40],[190,90,47],[175,64,39],[177,40,40],
     [159,55,37],[177,75,42],[181,85,43]]
Y = ['male', 'female', 'female', 'female','male', 'male','male',
     'female', 'male', 'female', 'male']

##Training the models
#Decision Tree
DTclf = tree.DecisionTreeClassifier()
DTclf = DTclf.fit(X,Y)

#Nearest Neighbour
NNclf = KNeighborsClassifier()
NNclf = NNclf.fit(X, Y)

#Guassian Process
GPclf = GaussianProcessClassifier()
GPclf = GPclf.fit(X, Y)

#Test using the same data
DTpred = DTclf.predict(X)
DTacc = accuracy_score(Y, DTpred)*100
print("Accuracy for Decision Tree: %s"% DTacc)

NNpred = NNclf.predict(X)
NNacc = accuracy_score(Y, NNpred)*100
print("Accuracy for Nearest Neighbour: %s" % NNacc)

GPpred = GPclf.predict(X)
GPacc = accuracy_score(Y, GPpred)*100
print("Accuracy for Gaussian Process: %s" % GPacc)

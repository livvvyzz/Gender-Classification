'''
Created on 31/07/2017

@author: olivia
'''
'''
Created on 31/07/2017

@author: olivia
'''
from sklearn import tree

#[height, weight, shoe size]
X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],
     [166,65,40],[190,90,47],[175,64,39],[177,40,40],
     [159,55,37],[177,75,42],[181,85,43]]
Y = ['male', 'female', 'female', 'female','male', 'male','male',
     'female', 'male', 'female', 'male']

#Classifiers

#Decision Tree
DTclf = tree.DecisionTreeClassifier()
DTclf = DTclf.fit(X,Y)

prediction = DTclf.predict([[190,70,43]])

print(prediction)
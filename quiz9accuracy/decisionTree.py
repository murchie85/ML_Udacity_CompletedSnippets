import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

#################################################################################


########################## DECISION TREE #################################



#### your code goes here
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

# make a prediction on test data
output = clf.predict(features_test)

import numpy as np
from sklearn.metrics import accuracy_score

acc = accuracy_score(labels_test, output)


### be sure to compute the accuracy on the test set



def submitAccuracies():
    return {"acc": round(acc, 3)}


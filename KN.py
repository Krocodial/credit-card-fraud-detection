import pandas as pd #for data processing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import preprocessing
import validation

#get data

data = np.loadtxt(fname='creditcard.csv', delimiter=',', skiprows=1, converters={30: lambda x: x[1:-1]})
rows = data.size

print(data.data.shape)


spliced_data = data[:, :30]
spliced_target = data[:, 30]

print(spliced_data.shape, spliced_target.shape)

#print(spliced_data)
#print(spliced_target)


xTrain, xTest, yTrain, yTest = model_selection.train_test_split(spliced_data, spliced_target, test_size = 0.2, random_state=0)

# applying smote
preprocessing.apply_smote(xTrain, yTrain)

print(xTrain.shape, yTrain.shape)


print("Data loaded...")
print("Training data")
print(xTrain)
print(yTrain)

print("Verification data")
print(xTest)
print(yTest)

print("length of x" + str(len(xTrain)))
print("length of y" + str(len(yTrain)))

print("Creating classifiers...")

clf = KNeighborsClassifier()

# Validation
validation.cross_validate(clf, xTrain, yTrain)

clf.fit(xTrain, yTrain)

print("KNeighborsClassifier")
score = clf.score(xTest, yTest)
print(str(score))

# Test
validation.test(clf, xTest, yTest)


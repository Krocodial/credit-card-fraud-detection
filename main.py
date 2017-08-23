from sklearn import tree, model_selection
import numpy as np

data = np.loadtxt(fname='creditcard.csv', delimiter=',', skiprows=1, converters={30: lambda x: x[1:-1]})
rows = data.size

X = data[:, :-1]
Y = data[:, -1]

xTrain, yTrain, xTest, yTest = model_selection.train_test_split(X, Y, test_size=0.4, random_state=1)


print("Data loaded...")
print("Training data")
print(xTrain)
print(yTrain)

print("Verification data")
print(xTest)
print(yTest)

print("Creating classifiers...")

clf = tree.DecisionTreeClassifier()
clf.fit(xTrain, yTrain)

print("Classifier loaded")

score = clf.score(xTest, yTest)
print("Decision Tree Score: " + str(score))

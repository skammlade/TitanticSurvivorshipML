import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing

trainFile = "../rawdata/train.csv"
testFile = "../rawdata/test.csv"
trainData = pd.read_csv(trainFile, header = 0)
testData = pd.read_csv(testFile, header = 0)
featureValues = trainData.loc[: , ["Pclass", "Sex", "Age"]]
targetValues = trainData.loc[: , "Survived"]
testFeatureValues = testData.loc[: , ["Pclass", "Sex", "Age"]]

le = preprocessing.LabelEncoder()
featureValues["Sex"] = featureValues["Sex"].str.replace("female", "0")
featureValues["Sex"] = featureValues["Sex"].str.replace("male", "1")
featureValues["Age"] = featureValues["Age"].fillna(0)
#print(featureValues)
clf = DecisionTreeClassifier()
clf = clf.fit(featureValues, targetValues)

testFeatureValues["Sex"] = testFeatureValues["Sex"].str.replace("female", "0")
testFeatureValues["Sex"] = testFeatureValues["Sex"].str.replace("male", "1")
testFeatureValues["Age"] = testFeatureValues["Age"].fillna(0)

#print(testFeatureValues)
predictedSurvival = clf.predict(testFeatureValues)
testData["Survived"] = predictedSurvival
print(testData.loc[:, ["PassengerId", "Survived"]])
testData.loc[:, ["PassengerId", "Survived"]].to_csv('willSKLearnTree01Output.csv', index=False)
#print(clf.predict(testFeatureValues))
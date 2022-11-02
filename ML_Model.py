"""
A new disease has been detected in rabbits. This disease is very rare, and does not occur
frequently. It is a genetic disease and can be detected in the different compounds that
are found in blood. The values of the compounds of blood, as well as the presence of the
disease in the rabbit are given in the dataset. Build a model that can (to an extent)
correctly predict the presence of the disease in the rabbit, given the compounds’ values.
Use various methods to judge and visualize the performance of the model.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
import seaborn as sns
import os

#create the ML model, train it, save the results!
os.remove("E:\\others\\gdsc\\prediction.csv")
dataSet = pd.read_csv('E:\\others\\gdsc\\rabbit_disease.csv')
training_data, testing_data = train_test_split(dataSet, test_size=0.2, random_state=25)
dataInput = training_data.iloc[:,:-1]
dataOutput = training_data.iloc[:,-1]

predictor = LogisticRegression(n_jobs=-1)
predictor.fit(X=dataInput, y=dataOutput)

testInput = testing_data.iloc[:,:-1]
output = list(predictor.predict(testInput))
expOutput = testing_data.iloc[:,-1:]

outputData = pd.DataFrame(expOutput)
outputData["Prediction"] = output
outputData["Difference"] = outputData["hasDisease"] - outputData["Prediction"]
outputData.to_csv("E:\\others\\gdsc\\prediction.csv")
print("Model trained! Results are tabulated!")

#Reads data from the resultant dataset
predDataSet = pd.read_csv("E:\\others\\gdsc\\prediction.csv")
resValue = predDataSet["Difference"].values
preValue = predDataSet["Prediction"].values
y_true, y_pred = predDataSet["hasDisease"].values, preValue
print("Data retrieved!")


#confusion matrix
#creates the confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
#seaborn heatmap
ax = sns.heatmap(cm, annot=True, cmap='flare')
ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

#Accuracy
acc = (tn + tp)/(tn + tp + fn + fp)
print("The accuracy is")
print("{} %".format(acc*100))
print()

#false positive rate
fpr = fp/(tn+fp)
print("The false positive rate is")
print("{} %".format(fpr*100))
print()


#false negative rate
fnr = fn/(tp+fn)
print("The false negative rate is")
print("{} %".format(fnr*100))
print()


#true negative rate
tnr = tn/(tn+fp)
print("The true negative rate is")
print("{} %".format(tnr*100))
print()


#true positive rate
tpr = tp/(tp+fn)
print("The true positive rate is")
print("{} %".format(tpr*100))

#log loss , dont understand?
loss = log_loss(y_true, y_pred)
print("The log loss is")
print("{} %".format(loss*100))

#Brier score, (y_pred - y_true)^2, how deviated is the predicted value from the true value
brier_loss = brier_score_loss(y_true, y_pred)
print("The brier_loss is")
print("{} %".format(brier_loss*100))

#Testing the model on a given input
testingInput = []
momentOfTruth = predictor.predict(testingInput)
print(momentOfTruth)

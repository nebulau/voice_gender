import imp
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import csv


input_data = pd.read_csv('VoiceGenderTrain.csv')
test_data = pd.read_csv('VoiceGenderTest.csv')

cols = [c for c in input_data.columns if c not in ['label']]
train = input_data.iloc[0 :2200]

test_cols = [tc for tc in test_data.columns]
test = test_data.iloc[0:900]
test_X =test[test_cols][0:900]

X = train[cols][0:2200]
Y = train['label'][0:2200]


seed = 8
test_size = 0
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()
   


    
model.fit(X_train, y_train)


predictions = model.predict(test_X)
dataframe = pd.DataFrame(predictions)
#for data in predictions:
    #writer.writerow(str(data))
dataframe.to_csv("results.csv", index=False, sep = ',',header = None)
print("Finished.")
#print(predictions)
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

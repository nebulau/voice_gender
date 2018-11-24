import imp
from numpy import loadtxt
#利用函数参数来设置模型参数，使用sklearn接口
#XGBClassifier是xgboost的sklearn包
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
test_size = 0 #拆分比例
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier(
    silent = 1,#打印运行消息default = 0
    max_child_weight = 1,#孩子节点最小样本权重和default = 1
    max_depth = 10,#定义一棵树的最大深度default = 6
    gamma = 0,#指定了分割时所需的最小损失的减少量default = 0
    max_delta_step = 0,#在逻辑回归中，当类别比例非常不平衡才起作用
    subsample = 1,#较低的值使算法比较保守，防止过拟合
    alpha = 0,#维数较高时使用，加快运行速度
    seed = 0,#种子随机数，etc其他参数
    )
   


    
model.fit(X_train, y_train)


predictions = model.predict(test_X)
#predictions = [value for value in model.predict(X_test)]
dataframe = pd.DataFrame(predictions)

dataframe.to_csv("results.csv", index=False, sep = ',',header = None)
print("Finished.")
#print(predictions)
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

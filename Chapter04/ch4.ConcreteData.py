import pandas as pd

HDNames= ['Cement','BFS','FLA','Water','SP','CA','FA','Age','CCS']
Data = pd.read_excel('ConcreteData.xlsx', names=HDNames)


print(Data.head(20))
print(Data.info())
summary = Data.describe()
print(summary)


# seaborn 绘图 更加美观 也更好用
import seaborn as sns
sns.set(style="ticks")
sns.boxplot(data = Data)

sns.pairplot(data = Data)

#归一化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
print(scaler.fit(Data))
DataScaled = scaler.fit_transform(Data)
DataScaled = pd.DataFrame(DataScaled, columns=HDNames)

summary = DataScaled.describe()
print(summary)

sns.boxplot(data = DataScaled)


# 数据集切割
from sklearn.model_selection import train_test_split

Predictors = pd.DataFrame(DataScaled.iloc[:,:8])
Response = pd.DataFrame(DataScaled.iloc[:,8])

Pred_train, Pred_test, Resp_train, Resp_test = train_test_split(Predictors,Response, test_size = 0.30, random_state = 1)
print(Pred_train.shape)
print(Pred_test.shape)
print(Resp_train.shape)
print(Resp_test.shape)



# 建立模型
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy']) # adam 优化器， 自带学习率调节 ，mse 回归模型常见loss 
model.fit(Pred_train, Resp_train, epochs=1000, verbose=1)

model.summary()

Y_predKM = model.predict(Pred_test)

from sklearn.metrics import r2_score # 

print('Coefficient of determination of Keras Model')
print(r2_score(Resp_test, Y_predKM))


# 箱线图原理
# 去掉离群点

Q1 = DataScaled.quantile(0.25)
Q3 = DataScaled.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

DataScaledOut = DataScaled[~((DataScaled < (Q1 - 1.5 * IQR)) | (DataScaled > (Q3 + 1.5 * IQR))).any(axis=1)]
print(DataScaledOut.shape)



# 绘图
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(121)
sns.boxplot(data = DataScaled)
plt.subplot(122)
sns.boxplot(data = DataScaledOut)

Predictors2 = pd.DataFrame(DataScaledOut.iloc[:,:8])
Response2 = pd.DataFrame(DataScaledOut.iloc[:,8])

Pred_train2, Pred_test2, Resp_train2, Resp_test2 = train_test_split(Predictors2,Response2, test_size = 0.30, random_state = 1)
print(Pred_train2.shape)
print(Pred_test2.shape)
print(Resp_train2.shape)
print(Resp_test2.shape)



# 建立模型
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.fit(Pred_train2, Resp_train2, epochs=1000, verbose=1)

model.summary()

Y_predKM2 = model.predict(Pred_test2)

print('Coefficient of determination of Keras Model')
print(r2_score(Resp_test, Y_predKM))

print('Coefficient of determination of Keras Model without outlier')
print(r2_score(Resp_test2, Y_predKM2))


plt.figure(1)
plt.subplot(121)
plt.scatter(Resp_test,Y_predKM)
plt.plot([0, 1], [0, 1], linewidth=2)
plt.subplot(122)
plt.scatter(Resp_test2, Y_predKM2)
plt.plot([0, 1], [0, 1], linewidth=2)

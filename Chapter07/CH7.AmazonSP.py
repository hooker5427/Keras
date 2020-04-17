import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



'''
股票预测
'''

np.random.seed(3)

Data = pd.read_csv('AMZN.csv',header=0, usecols=['Date','Close'],parse_dates=True,index_col='Date')

print(Data.info())
print(Data.head())
print(Data.describe())

plt.figure(figsize=(10,5))
plt.plot(Data)
plt.show()

DataPCh = Data.pct_change()

LogReturns = np.log(1 + DataPCh) 
print(LogReturns.tail(10))

plt.figure(figsize=(10,5))
plt.plot(LogReturns)
plt.show()


#归一化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

DataScaled = scaler.fit_transform(Data)

TrainLen = int(len(DataScaled) * 0.70)
TestLen = len(DataScaled) - TrainLen
TrainData = DataScaled[0:TrainLen,:]
TestData = DataScaled[TrainLen:len(DataScaled),:]

print(len(TrainData), len(TestData))



# 构建数据集
# TimeStep 时间步 
def DatasetCreation(dataset, TimeStep=1):
    DataX, DataY = [], []
    for i in range(len(dataset)- TimeStep -1):
            a = dataset[i:(i+ TimeStep), 0]
            DataX.append(a)
            DataY.append(dataset[i + TimeStep, 0])
    return np.array(DataX), np.array(DataY)



TimeStep = 1
TrainX, TrainY = DatasetCreation(TrainData, TimeStep)
TestX, TestY = DatasetCreation(TestData, TimeStep)

# reshape 成 batch_size  ,time_step , ndim 
TrainX = np.reshape(TrainX, (TrainX.shape[0], 1, TrainX.shape[1]))
TestX = np.reshape(TestX, (TestX.shape[0], 1, TestX.shape[1]))



# keras 模型
from keras.models import Sequential
from keras.layers import LSTM 
from keras.layers import Dense
from tensorflow import set_random_seed
set_random_seed(3)

model = Sequential()
model.add(LSTM(256, input_shape=(1, TimeStep)))  # lstm  ，当你需要使用LSTM的状态信息，请显时的指定batch_size采用batch_input_shape  ,return_state = True 

model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

model.fit(TrainX, TrainY, epochs=10, batch_size=1, verbose=1)

model.summary()

score = model.evaluate(TrainX, TrainY, verbose=0)
print('Keras Model Loss = ',score[0])
print('Keras Model Accuracy = ',score[1])

TrainPred = model.predict(TrainX)
TestPred = model.predict(TestX)

TrainPred = scaler.inverse_transform(TrainPred)
TrainY = scaler.inverse_transform([TrainY])
TestPred = scaler.inverse_transform(TestPred)
TestY = scaler.inverse_transform([TestY])

TrainPredictPlot = np.empty_like(DataScaled)
TrainPredictPlot[:, :] = np.nan
TrainPredictPlot[1:len(TrainPred)+1, :] = TrainPred

TestPredictPlot = np.empty_like(DataScaled)
TestPredictPlot[:, :] = np.nan
TestPredictPlot[len(TrainPred)+(1*2)+1:len(DataScaled)-1, :] = TestPred

plt.plot(scaler.inverse_transform(DataScaled))
plt.plot(TrainPredictPlot)
plt.plot(TestPredictPlot)
plt.show()

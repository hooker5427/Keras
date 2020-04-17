from keras.models import Sequential        # 顺序模型 
from keras.layers import Dense, Activation # 全连接层 ， 激活函数层 

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))  #  第一层的建立方式 ，也可以通过Input， 也可以通过参数的形式
model.add(Dense(1, activation='sigmoid')) # 激活函数 sigmiod ,将输入映射成为概率
model.compile(optimizer='rmsprop',         #优化器
              loss='binary_crossentropy', #  损失函数
              metrics=['accuracy']) #  评价指标， 这里是accuracy ，也就是准确度


# 一个好的习惯就是通过一些模型的参数来观看自己设计模型shape 是否和自己的预期一致
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))


# 训练
# data ->x 
# lable ->y 
# epochs 每一轮data比那里结束称为一个epoch
# batch_size  采用mini_batch的方式进行梯度更新， 32样本为一个batch 

model.fit(data, labels, epochs=10, batch_size=32)
#模型参数 结构预览
model.summary()


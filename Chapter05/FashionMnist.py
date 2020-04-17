from keras.datasets import fashion_mnist # 数据集 


# 加载数据
(XTrain, YTrain), (XTest, YTest) = fashion_mnist.load_data()

# 打印shape
print("X train shape:", XTrain.shape, "Y train shape:", YTrain.shape)
print("X test shape:", XTest.shape, "Y test shape:", YTest.shape)

# 绘图
import matplotlib.pyplot as plt
plt.imshow(XTrain[5])

print(YTrain[5])

LabelData = {
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'AnkleBoot',
}

plt.imshow(XTrain[5])
plt.title("(Object: " + str(LabelData[YTrain[5]]) + ")")

import numpy as np
unique, counts = np.unique(YTrain, return_counts=True) # 也可以用nuinque 求count 
print (dict(zip(unique, counts)))

unique, counts = np.unique(YTest, return_counts=True)
print (dict(zip(unique, counts)))


# 数据归一化
XTrain = XTrain / 255.0
XTest = XTest / 255.0


# 展示20张图片
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(XTrain[i], cmap=plt.cm.binary)
    plt.xlabel(LabelData[YTrain[i]])

# 模型建立
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Flatten, Activation 
from keras.layers import Dense, MaxPool2D, Conv2D # Dense全连接层,MaxPooling最大池化层,二维卷积层
from keras.initializers import Constant  # 常量初始化 

#  转化层Conv2D的数据合适
#  batch_size , channel, height , width 
XTrain = XTrain.reshape(XTrain.shape[0], 1, 28, 28)
XTest = XTest.reshape(XTest.shape[0], 1, 28, 28)
YTrain = np_utils.to_categorical(YTrain, 10)   # one-hot编码 
YTest = np_utils.to_categorical(YTest, 10)



# 建立keras模型
CNNModel = Sequential()



# conv2d 输出维度计算
# out_height  = ( height - kernel_size + 2 *padding  )/stride + 1 
# out_width  = ( width  - kernel_size + 2 *padding  )/stride + 1 

CNNModel.add(Conv2D(32,               # 卷积核的数量
                kernel_size=(2, 2),   #  卷积核的大小
                padding='same',       # padding 层  same 表示这一层输出与输出一致 
                activation='relu',     #非线性激活层，  relu (x ) = max( 0,x ) 
                bias_initializer=Constant(0.02),  #  bias 初始化， 一般为0也行
                kernel_initializer='random_uniform',   # kernel参数初始化  uniform ,normal , hekaiming...
                input_shape=(1, 28, 28)   # 指定输入层
            )
        )

CNNModel.add(Activation('relu'))

CNNModel.add(MaxPool2D(padding='same'))       # 最大池化
 
CNNModel.add(Conv2D(64,kernel_size=(2, 2), 
        padding='same', 
        bias_initializer=Constant(0.02), 
        kernel_initializer='random_uniform'
    )
)

CNNModel.add(Activation('relu'))

CNNModel.add(MaxPool2D(padding='same'))

CNNModel.add(Flatten())   # 一般进行展平之后进行全连接层的连接

CNNModel.add(Dense(128,
        activation='relu',
        bias_initializer=Constant(0.02), 
        kernel_initializer='random_uniform',         
    )
)

CNNModel.add(Dense(10, activation='softmax')) # 输出层， 10分类 ，经过softmax计算后映射层数据分类的概率，一般讲没有进过softmax 的最后一层的输出计为logits 
CNNModel.summary() 


#  多分类采用交叉熵进行loss的评估 ,交叉熵本质上衡量两种分布的接近程度
#  注意一点 ， 如果label 已经进行one_hot编码之后， loss采用 categorical_crossentropy
#  如果label 是整数类别， 比如(0 ,1,2,3, 4 ..... )等 , loss采用 sparse_categorical_crossentropy

CNNModel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
CNNModel.fit(XTrain, YTrain, epochs=1000,batch_size=32, verbose=1) # 训练

CNNModel.evaluate(XTest,YTest) # 评估

Scores = CNNModel.evaluate(XTest,YTest, verbose=1)
print('Test loss:', Scores[0])
print('Test accuracy:', Scores[1])

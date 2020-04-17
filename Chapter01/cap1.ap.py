from keras.layers import Input, Dense   # 导入输入层 ，全连接层
from keras.models import Model   # 模型

# 建立模型 
# 函数式api 
InputTensor = Input(shape=(100,))
H1 = Dense(10, activation='relu')( InputTensor)
H2 = Dense(20, activation='relu')(H1)
Output = Dense(1, activation='softmax')(H2)


model = Model(inputs=InputTensor, outputs=Output)

# 打印模型参数结果， 形状结构
model.summary()

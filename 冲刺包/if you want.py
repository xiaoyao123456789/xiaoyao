#
# 1.导入循环神经网络对应的相关包。
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,TimeDistributed
# 2.加载数据char_x= "do you love me"
#          char_y= "yes,me toolove"
sampel='do you love me yes,me toolove'
char_x='do you love me'
char_y='yes,me toolove'
# 3.数据的预处理(草参数设置)
char_set=list(set(sampel))

char_dict={w:i for i,w in enumerate(char_set)}

data_x=[char_dict[c] for c in char_x]
data_y=[char_dict[c] for c in char_y]

data=num_classes=len(char_set)
timestep=len(char_x)

data_x=tf.keras.utils.to_categorical(data_x,num_classes).reshape(-1,timestep,data)
y_one=tf.keras.utils.to_categorical(data_y,num_classes).reshape(-1,timestep,data)
# 4.模型的创建(LSTM)
model=Sequential()
model.add(LSTM(num_classes,input_shape=(timestep,data),return_sequences=True))
model.add(LSTM(num_classes,return_sequences=True))
model.add(TimeDistributed(tf.keras.layers.Dense(num_classes)))

model.summary()
# 5.模型配置(complie)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['acc'])
# 6.模型训练
h=model.fit(data_x,y_one,epochs=1000)
# 7.打印出损失值。
print(h.history['loss'])
predictions=model.predict(data_x)
print(predictions)
# 8.输出预测值和真实值。
for i,prediction in enumerate(predictions):
    index_x=tf.argmax(data_x[i],1)

    x_str=[char_set[j] for j in index_x]

    print(index_x,''.join(x_str))

    pre=tf.argmax(prediction,1)
    pre_str=[char_set[j] for j in pre]
    print(pre,''.join(pre_str))

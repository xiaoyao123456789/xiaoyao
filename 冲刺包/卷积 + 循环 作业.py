import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import  layers
import matplotlib.pyplot as plt

# def load_image(img_path,size=(32,32)):
lables = []
images = []

def readimg(file):
    img = plt.imread(file)
    img = img/255
    return img

path = r'D:\大D\ccc\深度学习（2）\data3'
for i in os.listdir(path):
    img = readimg(path+'\\'+i)
    images.append(img)
    lables.append(tf.keras.utils.to_categorical(int(i[0]),num_classes=3))
    print(images)
    print(lables)
images = np.array(images)
print(images)
labels = np.array(lables)
print(lables)

m = images.shape[0]
order = np.random.permutation(m)
imgArr = images[order]
y_one_hot = labels[order]
x_train,x_test,y_train,y_test = train_test_split(images,labels,train_size=0.7)

print(x_train.shape)
print(x_test.shape)

class LeNet5(tf.keras.Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = layers.Conv2D(6,[5,5],strides=1,padding='same')
        self.pool1 = layers.MaxPooling2D([2,2],strides=2)
        self.conv2 = layers.Conv2D(16,[5,5],strides=1,padding='valid')
        self.pool2 = layers.MaxPooling2D([2,2],strides=2)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=120,activation=tf.nn.relu)
        self.dense2 = layers.Dense(units=84,activation=tf.nn.relu)
        self.dense3 = layers.Dense(units=3,activation=tf.nn.softmax)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pool1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
model = LeNet5()
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-3),
    loss = tf.losses.categorical_crossentropy,
    metrics=['accuracy']
)
model.fit(x_train,y_train,epochs=100,batch_size=10,validation_data=(x_test,y_test),verbose=1)
score = model.evaluate(x_test,y_test,verbose=0)
print('Test score:', score[0]) # 代价
print('Test accuracy:', score[1]) # 准确率
#


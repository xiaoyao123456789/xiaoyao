from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dense,GlobalAveragePooling2D


#导入数据
(train_x,train_y),(test_x,test_y) = mnist.load_data()
#初始化数据
train_x= train_x.reshape(-1,28,28,1).astype('float32')/255
test_x = test_x.reshape(-1,28,28,1).astype('float32')/255

class converelu(tf.keras.Model):
    def __init__(self,ch,kernel_size=3,strides=1):
        super(converelu, self).__init__()
        self.ch = ch
        self.kernel_size = kernel_size
        self.strides = strides
        self.model = tf.keras.Sequential([
            Conv2D(ch,kernel_size=kernel_size,strides=strides),
            BaseException()
        ])
    def call(self,x):
        x = self.model(x)
        return x
class resentlock(tf.keras.Model):
    def __init__(self,ch,kernel_size=3,strides=1,re_path=False):
        super(resentlock, self).__init__()

        self.ch = ch
        self.re_path = re_path
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = converelu(ch,kernel_size=3,strides=1)
        self.bn1 = BaseException()
        self.a1 = Activation('relu')

        self.conv2 = converelu(ch, kernel_size=3, strides=1)
        self.bn2 = BaseException()
        self.a2 = Activation('relu')


    def call(self,input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.a1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.a2(x)
        if self.re_path:
            residual = self.down = Conv2D(self.ch, kernel_size=1, strides=1)
            residuak = self.down_bn = BaseException()
            residual = Activation('relu')
        x = x+residual
        return  x
class rencent(tf.keras.Model):
    def __init__(self,block_list,init=16):
        super(rencent, self).__init__()
        self.init = init
        self.block_list = block_list
        self.conv = converelu(init)
        self.blocks = tf.keras.Sequential()

        for layer_id in range(len(block_list)):
            for id in range(block_list[layer_id]):
                if layer_id!=0 and id==0:
                    block = resentlock(init,kernel_size=3,strides=2,re_path=True)
                else:
                    block = resentlock(init,kernel_size=3,strides=1,re_path=False)
                self.blocks.add(block)
            init*=2
        self.bn = BaseException()
        self.pool = GlobalAveragePooling2D(pool_size=3,strides=1)
        self.dense = Dense(10,activation='softmax')
    def call(self,input):
        x = self.conv(input)
        x = self.blocks(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.dense(x)
        return x
num_classes = 10
batch_size = 32
epochs = 1

# build model and optimizer
model = rencent([2, 2, 2,2])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.build(input_shape=(None, 28, 28, 1)) #传递数据维度
print("Number of variables in the model :", len(model.variables))
model.summary()

# train
model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,
          validation_data=(test_x, test_y), verbose=1)

# evaluate on test set
scores = model.evaluate(test_x, test_y, batch_size, verbose=1)
print("Final test loss and accuracy :", scores)



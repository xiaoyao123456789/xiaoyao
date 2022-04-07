
import tensorflow as tf
import numpy as np
from tensorflow.keras import models,layers

(train_x,train_y),(test_x,test_y)=tf.keras.datasets.cifar10.load_data()
print(type(train_x))
train_x=train_x.reshape(-1,32,32,3).astype('float32')/255
test_x=test_x.reshape(-1,32,32,3).astype('float32')/255
train_y=tf.keras.utils.to_categorical(train_y,10)
test_y=tf.keras.utils.to_categorical(test_y,10)
print(train_x.shape)
print(test_x.shape)
train_db=tf.data.Dataset.from_tensor_slices((train_x,train_y)).shuffle(50000).batch(256)
test_db=tf.data.Dataset.from_tensor_slices((test_x,test_y)).shuffle(10000).batch(256)

class ResnetBlk(tf.keras.Model):
    def __init__(self,ch,strides=1,residual_path=False):
        super(ResnetBlk, self).__init__()
        self.residual_path=residual_path
        self.ch = ch
        self.strides = strides

        self.c1=layers.Conv2D(ch,3,strides=strides,padding='same',use_bias=False)
        self.b1=layers.BatchNormalization()
        self.a1=layers.Activation('relu')

        self.c2=layers.Conv2D(ch,3,strides=1,padding='same',use_bias=False)
        self.b2=layers.BatchNormalization()

        if residual_path:
            self.down_c1=layers.Conv2D(ch,1,strides=strides,padding='same',use_bias=False)
            self.down_b1=layers.BatchNormalization()

    def call(self, inputs, training=None):
        residual=inputs


        x=self.c1(inputs)
        x = self.b1(x)
        x=self.a1(x)
        x=self.b2(x)
        x=self.c2(x)

        if self.residual_path:

            residual=self.down_c1(inputs)
            residual = self.down_b1(residual)

        x=self.a2(x)
        x=x+residual
        return x

class Resnet(tf.keras.Model):
    def __init__(self,block_list,num_classes,init_ch=16,**kwargs):
        super(Resnet, self).__init__(**kwargs)
        self.out_ch=init_ch

        self.c1=layers.Conv2D(self.out_ch,3,strides=1,padding='same',
                              kernel_initializer=tf.random_normal_initializer(),
                              use_bias=False)
        self.b1=layers.BatchNormalization()
        self.a1=layers.Activation('relu')

        self.blocks=models.Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id!=0 and layer_id==0:
                    block=ResnetBlk(self.out_ch,strides=2,residual_path=True)
                else:
                    block=ResnetBlk(self.out_ch,residual_path=False)
                self.blocks.add(block)
            self.out_ch *= 2
        self.b2=layers.BatchNormalization()
        self.a2=layers.Activation('relu')
        self.p1=layers.GlobalAveragePooling2D()
        self.f1=layers.Dense(num_classes,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x=self.c1(inputs)

        x=self.blocks(x)
        x = self.b2(x)
        x = self.a2(x)
        x=self.p1(x)
        x=self.f1(x)
        return x

def main():
    model=Resnet([2,2,2,2],10)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=False),
                  metrics = ["accuracy"])



    model.fit(train_db,epochs=1,validation_data=test_db)

    model.summary()

    score=model.evaluate(test_db)
    print(score)

if __name__ == '__main__':
    main()

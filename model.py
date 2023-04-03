#tF 2.6.0
import tensorflow as tf
#LAYERS, REGULARIZERS, MODELS, CALLBACKS
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *


class DNN:
    
    def __init__(self, input_shape = None):
        self.input_shape = input_shape

    def bn_relu_conv(self,X, nb_filter, stride_rate, padding_typ, dilation):

        #Batch Normalization
        X = BatchNormalization()(X)
        #ReLU
        X = Activation('relu')(X)
        #Convolution
        X = Conv2D(filters = nb_filter, 
               kernel_size = (3,3),
               strides = stride_rate,
               padding = padding_typ,
               dilation_rate = dilation,
               activation = None,
               kernel_initializer = 'he_normal',
               kernel_regularizer = L2(l2 = 1.e-4))(X)
    
        return X

    def RB(self,X, nb_filter,unequal_channels):

    
        conv0 = self.bn_relu_conv(X,nb_filter, (1,1), 'same', (1,1))
        conv1 = self.bn_relu_conv(conv0, nb_filter, (1,1), 'same', (1,1))
    
        #unequal_channels= tf.shape(skip_connection)[1]!=tf.shape(X)[1]
    
        if unequal_channels:
            skip_connection = Conv2D(filters = nb_filter, 
                                 kernel_size = (1,1),
                                 strides = (1,1),
                                 padding = 'valid',
                                 kernel_initializer = 'he_normal',
                                 kernel_regularizer = tf.keras.regularizers.L2(l2 = 1.e-4))(X)
        else:
            skip_connection = X
    
        conv2 = Add()([conv1, skip_connection])
    
        return conv2

    def DRB(self,X, nb_filter):
    
        conv0 = self.bn_relu_conv(X,     nb_filter, (2,2), 'same', (1,1))
        conv1 = self.bn_relu_conv(conv0, nb_filter, (1,1), 'same', (1,1))
    
        skip_connection = Conv2D(filters = nb_filter, 
                                 kernel_size = (1,1),
                                 strides = (2,2),
                                 padding = 'valid',
                                 kernel_initializer = 'he_normal',
                                 kernel_regularizer = tf.keras.regularizers.L2(l2 = 1.e-4))(X)
    
        conv2 = Add()([conv1, skip_connection])
    
        conv3 = self.RB(conv2, nb_filter, False)
    
        return conv3

    def DiRB(self, X, nb_filter):
    
        conv0 = self.bn_relu_conv(X,nb_filter, (2,2), 'same', (1,1))
        conv1 = self.bn_relu_conv(conv0, nb_filter, (1,1), 'same', (2,2))
    
        skip_connection = Conv2D(filters = nb_filter, 
                                 kernel_size = (1,1),
                                 strides = (2,2),
                                 padding = 'valid',
                                 kernel_initializer = 'he_normal',
                                 kernel_regularizer = tf.keras.regularizers.L2(l2 = 1.e-4))(X)
    
        conv2 = Add()([conv1, skip_connection])
    
        conv3 = self.bn_relu_conv(conv2, nb_filter, (1,1), 'same', (2,2))
        conv4 = self.bn_relu_conv(conv3, nb_filter, (1,1), 'same', (2,2))
    
        #skip_connection = Conv2D(filters = nb_filter, 
        #                             kernel_size = (1,1),
        #                             strides = (1,1),
        #                             padding = 'valid',
        #                             kernel_initializer = 'he_normal',
        #                             kernel_regularizer = tf.keras.regularizers.L2(l2 = 1.e-4))(conv2)
    
        conv5 = Add()([conv4, conv2])
    
        return conv5

    def URB(self, X, nb_filter):
    
        #Batch Normalization
        X = BatchNormalization()(X)
        #ReLU
        X = Activation('relu')(X)
        #Convolution Transpose
        conv0 = Conv2DTranspose(filters = nb_filter, 
                        kernel_size = (2,2),
                        strides = (2,2),
                        padding = 'same' ,
                        kernel_initializer = 'he_normal',
                        kernel_regularizer = tf.keras.regularizers.L2(l2 = 1.e-4))(X)
    
        conv1 = self.bn_relu_conv(conv0, nb_filter, (1,1), 'same', (1,1))
    
        skip_connection = Conv2DTranspose(filters = nb_filter, 
                                      kernel_size = (2,2),
                                      strides = (2,2),
                                      padding = 'valid',
                                      kernel_initializer = 'he_normal',
                                      kernel_regularizer = tf.keras.regularizers.L2(l2 = 1.e-4))(X)
    
        conv2 = Add()([conv1, skip_connection])
    
        conv3 = self.RB(conv2, nb_filter, False)
    
    
        return conv3

    def model_instance(self):

        d_rate = 0.05
    
        X_Input = Input(self.input_shape) #1024
    
        X_512 = self.DRB(X_Input, 16) #512
        X_512 = Dropout(d_rate)(X_512)
    
    
        X_256 = self.DiRB(X_512, 32) #256
        X_256 = Dropout(d_rate)(X_256)
        
        '''X_256 = RB(X_Input, 32, True)
        X_256 = Dropout(d_rate)(X_256)'''
    
    
        X_128 = self.DiRB(X_256, 48) #128
        X_128 = Dropout(d_rate)(X_128)
    
    
        X_64 = self.DiRB(X_128, 64) #64
        X_64 = Dropout(d_rate)(X_64)
    
    
        X_32 = self.DRB(X_64, 96) #32
        X_32 = Dropout(d_rate)(X_32)
    
    
        X_16 = self.DRB(X_32, 128) #16
        X_16 = Dropout(d_rate)(X_16)
    
    
        X_8 = self.DRB(X_16, 192) #8
        X_8 = Dropout(d_rate)(X_8)
    
        Y_16 = self.URB(X_8, 256) #16
        Y_16 = Dropout(d_rate)(Y_16)
        Y_16 = Concatenate(axis=3)([X_16, Y_16])
    
        Y_32 = self.URB(Y_16, 192) #32
        Y_32 = Dropout(d_rate)(Y_32)
        Y_32 = Concatenate(axis=3)([X_32, Y_32])
    
        Y_64 = self.URB(X_32, 128) #64
        Y_64 = Dropout(d_rate)(Y_64)
        Y_64 = Concatenate(axis=3)([X_64, Y_64])
    
        Y_128 = self.URB(Y_64, 96) #128
        Y_128 = Dropout(d_rate)(Y_128)
        Y_128 = Concatenate(axis=3)([X_128, Y_128])
    
        Y_256 = self.URB(Y_128, 64) #256
        Y_256 = Dropout(d_rate)(Y_256)
        Y_256 = Concatenate(axis=3)([X_256, Y_256])
    
        Y_512 = self.URB(Y_256, 32) #512
        Y_512 = Dropout(d_rate)(Y_512)
        Y_512 = Concatenate(axis=3)([X_512, Y_512])
        
        Y_1024 = self.URB(Y_512, 48) #1024
    
        Y_1024_16 = self.RB(Y_1024, 16, True)
    
        Y_1024_1 = self.RB(Y_1024_16, 1, True)
        
        Y_1024_1 = Add()([X_Input, Y_1024_1]) #adding input to final output

        Y = Activation('relu')(Y_1024_1)

        model = Model(inputs = X_Input, outputs = Y)

        return model


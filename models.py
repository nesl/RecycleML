import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *

# define network!
def pre_imu_model(input_x, input_y, reg = 0, 
                  high_layers = 'one', 
                  num_feat_map = 16, summary=False):  
    #print('building the model ... ')        
    model = Sequential()
    model.add(Conv2D(num_feat_map, kernel_size=(1, 5),
                 activation='relu',
                 input_shape=input_x.shape[1:],
                 padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(num_feat_map, kernel_size=(1, 5), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(128, name='DENSE_2'))
    if high_layers == 'two':
        if summary:
            print(model.summary())
        return model
    else:
        model.add(Activation('relu', name = 'ACT_2'))
        model.add(Dense(128, name='DENSE_1'))
        if summary:
            print(model.summary())
        return model

def full_imu_model(input_x, input_y, reg = 0, num_feat_map = 16, summary=False):  
    #print('building the model ... ')        
    model = Sequential()
    model.add(Conv2D(num_feat_map, kernel_size=(1, 5),
                 activation='relu',
                 input_shape=input_x.shape[1:],
                 padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(num_feat_map, kernel_size=(1, 5), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(128, name='DENSE_2'))
    model.add(Activation('relu', name = 'ACT_2'))
    model.add(Dense(128, name='DENSE_1'))
    model.add(Activation('relu', name = 'ACT_1'))

    model.add(Dense(input_y.shape[1], activation='softmax', name = 'SOFT'))
    
    if summary:
        print(model.summary())
    return model


def pre_sound_model(input_x, input_y, reg = 0, 
                    high_layers = 'one', 
                    num_feat_map = 16, summary=False):  
    """ Return the Keras model of the network
    """
    #print('building the model ... ')            
    model = Sequential()
    model.add(Dense(32, input_shape=(193,), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(596, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(128, name='DENSE_2'))
    if high_layers == 'two':
        if summary:
            print(model.summary())
        return model
    else:
        model.add(Activation('relu', name = 'ACT_2'))
        model.add(Dense(128, name='DENSE_1'))
        if summary:
            print(model.summary())
        return model

def full_sound_model(input_x, input_y, reg = 0, num_feat_map = 16, summary=False):  
    """ Return the Keras model of the network
    """
    #print('building the model ... ')            
    model = Sequential()
    model.add(Dense(32, input_shape=(193,), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(596, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(128, activation='relu'))
    
    model.add(Dense(128, name='DENSE_2'))
    model.add(Activation('relu', name = 'ACT_2'))
    model.add(Dense(128, name='DENSE_1'))
    model.add(Activation('relu', name = 'ACT_1'))
    
    model.add(Dense(input_y.shape[1], activation='softmax', name = 'SOFT'))
    if summary:
        print(model.summary())
    return model


def pre_video_model(input_x, input_y, reg = 0,
                    high_layers = 'one',  
                    num_feat_map = 16, summary=False):  

    #print('building the model ... ')            
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=input_x.shape[1:]))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool3' ,dim_ordering="th"))
    # 4th layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool4', dim_ordering="th"))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(1024,  name='fc6'))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    
    model.add(Dense(128, name='DENSE_2'))
    if high_layers == 'two':
        if summary:
            print(model.summary())
        return model
    else:
        model.add(Activation('relu', name = 'ACT_2'))
        model.add(Dense(128, name='DENSE_1'))
        if summary:
            print(model.summary())
        return model


def full_video_model(input_x, input_y, reg = 0, num_feat_map = 16, summary=False):  
    #print('building the model ... ')            
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=input_x.shape[1:]))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool3' ,dim_ordering="th"))
    # 4th layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool4', dim_ordering="th"))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(1024,  name='fc6'))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    
    model.add(Dense(128, name='DENSE_2'))
    model.add(Activation('relu', name = 'ACT_2'))
#     model.add(Dropout(.5))
    model.add(Dense(128, name='DENSE_1'))
    model.add(Activation('relu', name = 'ACT_1'))
    
    model.add(Dense(input_y.shape[1], activation='softmax', name = 'SOFT'))
    
    if summary:
        print(model.summary())
    return model

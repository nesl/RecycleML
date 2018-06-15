# random seed.
rand_seed = 1
from numpy.random import seed
seed(rand_seed)
from tensorflow import set_random_seed
set_random_seed(rand_seed)

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
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

from matplotlib import pyplot as plt
from PIL import Image as img_PIL

from utils import *
from models import *

np.random.rand(1) # array([0.417022])

CLASS_7=['downstair','upstair','run','jump','walk', 'handwashing', 'exercise']
CLASS_5=['run','jump','walk', 'handwashing', 'exercise']
##mode: imu, sound, video, video2sound, video2imu, imu2video, sound2video, imu2sound, sound2imu
mode = 'imu'

print(CLASS_7)

# testing dataset!
Test_data = np.load('../1Extract_all_data/extracted/Data_test_71.pkl')
Test_imu = Test_data[0]
Test_imu = Test_imu.reshape( [Test_imu.shape[0], 1, Test_imu.shape[1], Test_imu.shape[2]] )
Test_label = Test_data[1]
Test_label = one_hot_label(Test_label, CLASS_7)
print('\nClasses distribution: ',np.sum(Test_label, axis =0))
Test_sound = Test_data[2]
Test_video=Test_data[3]
Test_video = Test_video.transpose((0, 4, 1, 2, 3))
# shuffle dataset
Test_imu, Test_label, Test_sound, Test_video = shuffle_data(Test_imu, Test_label, 
                                                            Test_sound, Test_video )
print('\nDataset shuffled!')
print('Testing Data shape:')
print('IMU features: ',Test_imu.shape,
      '\nLabels: ',Test_label.shape,
      '\nSound features: ',Test_sound.shape,
      '\nVideo features: ',Test_video.shape)

# personalized dataset!
Personal_data = np.load('../1Extract_all_data/extracted/Data_personal_71.pkl')
Personal_imu = Personal_data[0]
Personal_imu = Personal_imu.reshape( [Personal_imu.shape[0], 1, Personal_imu.shape[1], Personal_imu.shape[2]] )
Personal_label = Personal_data[1]
Personal_label = one_hot_label(Personal_label, CLASS_5)
print('\nClasses distribution: ',np.sum(Personal_label, axis =0))
Personal_sound = Personal_data[2]
Personal_video=Personal_data[3]
Personal_video = Personal_video.transpose((0, 4, 1, 2, 3))
# shuffle dataset
Personal_imu, Personal_label, Personal_sound, Personal_video = shuffle_data(Personal_imu, 
                                                                            Personal_label, 
                                                                            Personal_sound, 
                                                                            Personal_video )
print('\nDataset shuffled!')
print('Personalized Data shape:')
print('IMU features: ',Personal_imu.shape,
      '\nLabels: ',Personal_label.shape,
      '\nSound features: ',Personal_sound.shape,
      '\nVideo features: ',Personal_video.shape)


# Training dataset!
Train_data = np.load('../1Extract_all_data/extracted/Data_train_all.npz')

Train_imu = Train_data['arr_0']
Train_imu = Train_imu.reshape( [Train_imu.shape[0], 1, Train_imu.shape[1], Train_imu.shape[2]] )
Train_label = Train_data['arr_1']
Train_label = one_hot_label(Train_label, CLASS_7)
print('\nClasses distribution: ',np.sum(Train_label, axis =0))
Train_sound = Train_data['arr_2']
Train_video=Train_data['arr_3']
Train_video = Train_video.transpose((0, 4, 1, 2, 3))
# shuffle dataset
Train0_imu, Train0_label, Train0_sound, Train0_video = shuffle_data(Train_imu, Train_label, 
                                                            Train_sound, Train_video )
print('Training Data shape:')
print('IMU features: ',Train0_imu.shape,
      '\nLabels: ',Train0_label.shape,
      '\nSound features: ',Train0_sound.shape,
      '\nVideo features: ',Train0_video.shape)

Train_num = 5000
Transfer_num = 6000

# Testing set, personalization set unchanged
# Training set is split into 3 parts: Train_data, Transfer_data, LimitTrain_data

Transfer_imu   = Train0_imu[Train_num: Train_num+Transfer_num]
Transfer_label = Train0_label[Train_num: Train_num+Transfer_num]
Transfer_sound = Train0_sound[Train_num: Train_num+Transfer_num]
Transfer_video = Train0_video[Train_num: Train_num+Transfer_num]

LimitTrain_imu   = Train0_imu[Train_num+Transfer_num:]
LimitTrain_label = Train0_label[Train_num+Transfer_num:]
LimitTrain_sound = Train0_sound[Train_num+Transfer_num:]
LimitTrain_video = Train0_video[Train_num+Transfer_num:]

Train_imu   = Train0_imu[0:Train_num]
Train_label = Train0_label[0:Train_num]
Train_sound = Train0_sound[0:Train_num]
Train_video = Train0_video[0:Train_num]

print('Train Data shape:')
print('Classes distribution: ',np.sum(Train_label, axis =0))
print('IMU features: ',Train_imu.shape,
      '\nLabels: ',Train_label.shape,
      '\nSound features: ',Train_sound.shape,
      '\nVideo features: ',Train_video.shape)
print('Transfer Data shape:')
print('Classes distribution: ',np.sum(Transfer_label, axis =0))

print('LimitTrain Data shape:')
print('Classes distribution: ',np.sum(LimitTrain_label, axis =0))

# training pre-trained model: training data is train data, testing data is testing set
if mode =='imu':
    ORG_NET = mode
    Train_data = Train_imu
    Test_data = Test_imu
elif mode == 'sound':
    ORG_NET = mode
    Train_data = Train_sound
    Test_data = Test_sound
elif mode == 'video':
    ORG_NET = mode
    Train_data = Train_video
    Test_data = Test_video

# knowledge transferring: 
# Train_data is 6k unlabeled data of org modal, test_data is only used for evaluating loaded model
# teacher_result is calculated latent feature using Train_data

# New_train_data is 6k unlabeled data for another modal
# New_test_data is testing data of new modal

# Limit_Train_data is the LimitTrain_data for new modal

elif mode == 'video2sound':
    ORG_NET = mode.split('2')[0]
    NEW_NET = mode.split('2')[1]
    Train_data = Transfer_video
    Test_data = Test_video
    New_Train_data = Transfer_sound
    New_Test_data = Test_sound
    Limit_Train_data = LimitTrain_sound
    Limit_Train_label = LimitTrain_label
    
elif mode == 'video2imu':
    ORG_NET = mode.split('2')[0]
    NEW_NET = mode.split('2')[1]
    Train_data = Transfer_video
    Test_data = Test_video
    New_Train_data = Transfer_imu
    New_Test_data = Test_imu
    Limit_Train_data = LimitTrain_imu
    Limit_Train_label = LimitTrain_label
    
elif mode == 'imu2video':
    ORG_NET = mode.split('2')[0]
    NEW_NET = mode.split('2')[1]
    Train_data = Transfer_imu
    Test_data = Test_imu
    New_Train_data = Transfer_video
    New_Test_data = Test_video
    Limit_Train_data = LimitTrain_video
    Limit_Train_label = LimitTrain_label
    
elif mode == 'sound2video':
    ORG_NET = mode.split('2')[0]
    NEW_NET = mode.split('2')[1]
    Train_data = Transfer_sound
    Test_data = Test_sound
    New_Train_data = Transfer_video
    New_Test_data = Test_video
    Limit_Train_data = LimitTrain_video
    Limit_Train_label = LimitTrain_label
    
elif mode == 'imu2sound':
    ORG_NET = mode.split('2')[0]
    NEW_NET = mode.split('2')[1]
    Train_data = Transfer_imu
    Test_data = Test_imu
    New_Train_data = Transfer_sound
    New_Test_data = Test_sound
    Limit_Train_data = LimitTrain_sound
    Limit_Train_label = LimitTrain_label
    
elif mode == 'sound2imu':
    ORG_NET = mode.split('2')[0]
    NEW_NET = mode.split('2')[1]
    Train_data = Transfer_sound
    Test_data = Test_sound
    New_Train_data = Transfer_imu
    New_Test_data = Test_imu
    Limit_Train_data = LimitTrain_imu
    Limit_Train_label = LimitTrain_label
    
else:
    print('Wrong mode!!!')

print('Mode is: ',mode)


reg = 0.1
Sound_model  = full_imu_model(Train_data, Train_label, summary=True)

batch_size = 128
epochs = 800

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
Sound_model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=sgd,
              optimizer= adam,
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model/0_'+ORG_NET+'.hdf5',
                               verbose=1,
                               monitor='val_acc',
                               save_best_only=True)

learning_hist = Sound_model.fit(Train_data, Train_label,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            shuffle=True,
                            callbacks=[checkpointer], 
                            validation_data=(Test_data, Test_label))


name = '0_bench_'+ORG_NET
plot_learning_curves(learning_hist, name )


from keras.models import load_model
model_load=load_model('model/0_'+ORG_NET+'.hdf5')

# model_load.summary()  
print(CLASS_7)
evaluate_model(model_load, Test_data, Test_label)
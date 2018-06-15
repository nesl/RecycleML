# random seed.
rand_seed = 1
from numpy.random import seed
seed(rand_seed)
from tensorflow import set_random_seed
set_random_seed(rand_seed)

import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.models import Model
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

# Load matplotlib images inline
%matplotlib inline

# These are important for reloading any code you write in external .py files.
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

np.random.rand(1) # array([0.417022])

CLASS_7=['downstair','upstair','run','jump','walk', 'handwashing', 'exercise']
CLASS_5=['run','jump','walk', 'handwashing', 'exercise']
##mode: imu, sound, video, video2sound, video2imu, imu2video, sound2video, imu2sound, sound2imu
mode = 'video2sound'
## high_layers:  'one', 'two'
high_layers = 'one'

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)


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

# train_data, transfer_data, limitTrain data
Transfer_data = np.load('data/Transfer_data.npz')
Transfer_imu = Transfer_data['arr_0']
Transfer_label = Transfer_data['arr_1']
Transfer_sound = Transfer_data['arr_2']
Transfer_video = Transfer_data['arr_3']

LimitTrain_data = np.load('data/LimitTrain_data.npz')
LimitTrain_imu = LimitTrain_data['arr_0']
LimitTrain_label = LimitTrain_data['arr_1']
LimitTrain_sound = LimitTrain_data['arr_2']
LimitTrain_video = LimitTrain_data['arr_3']

# print('Train Data shape:')
# print('Classes distribution: ',np.sum(Train_label, axis =0))
# print('IMU features: ',Train_imu.shape,
#       '\nLabels: ',Train_label.shape,
#       '\nSound features: ',Train_sound.shape,
#       '\nVideo features: ',Train_video.shape)
print('\nTransfer Data distribution:')
print('Num of samples: ', Transfer_label.shape[0])
print('Classes distribution: ',np.sum(Transfer_label, axis =0))

print('LimitTrain Data distribution:')
print('Num of samples: ', LimitTrain_label.shape[0])
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
saved_name = ORG_NET+'2'+NEW_NET
print('saved_name is: ',saved_name)


model_load=load_model('model/0_'+ORG_NET+'.hdf5')
model_load.summary()

evaluate_model(model_load, Test_data, Test_label)


if high_layers == 'one':
    teacher_model = Model(inputs=model_load.input,
                            outputs=model_load.get_layer('DENSE_1').output)
elif high_layer == 'two':
    teacher_model = Model(inputs=model_load.input,
                            outputs=model_load.get_layer('DENSE_2').output)
else:
    print('Wrong high_layer!')

teacher_result = teacher_model.predict(Train_data)
print('Data: ', Train_data.shape, teacher_result.shape)


reg_value = 0.1

if NEW_NET == 'sound':
    student_model = pre_sound_model(New_Train_data, teacher_result, summary=True)
elif NEW_NET == 'imu':
    student_model = pre_imu_model(New_Train_data, teacher_result, summary=True)
elif NEW_NET == 'video':
    student_model = pre_video_model(New_Train_data, teacher_result, summary=True)
else:
    print("wrong new net!")


# transfer training:
epochs = 500
batch_size = 128

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

student_model.compile(
                        loss='mse',  # 'kld'
#                       optimizer=sgd,
                      optimizer=adam,
                      metrics=['accuracy'])

file_name = 'model/'+saved_name+'_1_pre.hdf5'
checkpointer = ModelCheckpoint(filepath=file_name,
                               verbose=1,
                               monitor = 'val_loss',
                               save_best_only=True)

print('Model saved at: '+ file_name)
H_transfer = student_model.fit( New_Train_data, teacher_result,
                                batch_size=batch_size,
                    #             optimizer=sgd,
                                epochs=epochs,
                                verbose=1,
                                shuffle=True,
                                callbacks=[checkpointer], 
                                validation_data = [New_Train_data, teacher_result])


name = saved_name+'1_pre'
plot_learning_curves(H_transfer, name)


direct_student = load_model('model/'+saved_name+'_1_pre.hdf5')
if high_layers == 'one':
    direct_student.add( model_load.get_layer('ACT_1') )
    direct_student.add( model_load.get_layer('SOFT') )
elif high_layers == 'two':
    direct_student.add( model_load.get_layer('ACT_2') )
    direct_student.add( model_load.get_layer('DENSE_1') )
    direct_student.add( model_load.get_layer('ACT_1') )
    direct_student.add( model_load.get_layer('SOFT') )
    
direct_student.summary()


evaluate_model(direct_student, New_Test_data, Test_label)

personal_student = load_model('model/'+saved_name+'_1_pre.hdf5')
for layer in personal_student.layers:
     layer.trainable = False
        
if high_layers == 'one':
    personal_student.add( Activation('relu', name ='ACT_1'))
    personal_student.add( Dense(Test_label.shape[1], activation='softmax', name = 'SOFT')  )
elif high_layers == 'two':
    personal_student.add( Activation('relu',name ='ACT_2'))
    personal_student.add( Dense(128, activation='relu',name ='DENSE_ACT_1')  )
#     model.add(Dropout(0.5))?
    personal_student.add( Dense(Test_label.shape[1], activation='softmax',name ='SOFT')  )
    
personal_student.summary()


limit_size = 500

New_train_limit = Limit_Train_data[:limit_size]
Train_label_limit = Limit_Train_label[:limit_size]

epochs = 100
batch_size = 128

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
personal_student.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=sgd,
              optimizer= adam,
              metrics=['accuracy'])
file_name = 'model/'+saved_name+'_2_done.hdf5'
checkpointer = ModelCheckpoint(filepath=file_name,
                               verbose=1,
                               monitor = 'val_acc',
                               save_best_only=True)

print('Model saved at: '+file_name)
H_personal = personal_student.fit( New_train_limit, Train_label_limit,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    shuffle=True,
                                    callbacks=[checkpointer], 
                                    validation_data=(New_Test_data, Test_label))



name = saved_name+'2_personal'
plot_learning_curves(H_personal,name)
personal_student = load_model('model/'+saved_name+'_2_done.hdf5')
evaluate_model(personal_student, New_Test_data, Test_label)


reg_value = 0.1

if NEW_NET == 'sound':
    scratch_new_model = full_sound_model(New_train_limit, 
                                         Train_label_limit, summary=True)
elif NEW_NET == 'imu':
    scratch_new_model = full_imu_model(New_train_limit, 
                                       Train_label_limit, summary=True)
elif NEW_NET == 'video':
    scratch_new_model = full_video_model(New_train_limit, 
                                         Train_label_limit, summary=True)
else:
    print("wrong new net!")


epochs = 500
batch_size = 128
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
scratch_new_model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=sgd,
              optimizer= adam,
              metrics=['accuracy'])
file_name = 'model/'+saved_name+'_3_scratch.hdf5'
checkpointer = ModelCheckpoint(filepath=file_name,
                               verbose=1,
                               monitor = 'val_acc',
                               save_best_only=True)

print('Model saved at: '+file_name)
H_scratch = scratch_new_model.fit( New_train_limit, Train_label_limit,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    shuffle=True,
                                    callbacks=[checkpointer], 
                                    validation_data=(New_Test_data, Test_label))


name = saved_name+'3_scratch'
plot_learning_curves(H_scratch,name)
scratch_new_model = load_model('model/'+saved_name+'_3_scratch.hdf5')
evaluate_model(scratch_new_model, New_Test_data, Test_label)

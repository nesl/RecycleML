import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras import optimizers
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint


import itertools
from keras.preprocessing import image
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import glob
import os
import time

from matplotlib import pyplot as plt
from PIL import Image as img_PIL

from models import *

def evaluate_model(model_load, Test_data, Test_label):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score


    pred_data = Test_data
    pred_label = Test_label

    y_pred = np.argmax(model_load.predict(pred_data), axis=1)
    y_true = np.argmax(pred_label, axis=1)
    print('Accuracy on testing data:',sum(y_pred==y_true)/y_true.shape[0])
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    class_wise_f1 = np.round(f1_score(y_true, y_pred, average=None)*100)*0.01
    print('the mean-f1 score: {:.2f}'.format(np.mean(class_wise_f1)))

def plot_learning_curves(learning_hist, name = 'temp', no_fig=False):
    #print(single)
    learning_key = learning_hist.history.keys()
#     print( learning_key )
    metric = []
    for k in learning_key:
#         print(k)
        metric.append((k))
    train_loss = learning_hist.history[metric[2]]
    train_metric = learning_hist.history[metric[3]]
    val_loss = learning_hist.history[metric[0]]
    val_metric = learning_hist.history[metric[1]]
    
    # saving training data to history
    train_loss = np.array(train_loss).reshape( (len(train_loss),1) )
    train_metric = np.array(train_metric).reshape( (len(train_metric),1) )
    val_loss = np.array(val_loss).reshape( (len(val_loss),1) )
    val_metric = np.array(val_metric).reshape( (len(val_metric),1) )
    saved_data = np.concatenate([train_metric, val_metric, train_loss, val_loss], 1)
    np.savetxt('history/'+name+'_test.csv', saved_data)
    if name != 'temp':
        print('Learning History Saved: '+ 'history/'+name+'_test.csv')
    
    if no_fig:
        return
    epoch_num = np.arange(len(train_loss))
    width = 8
    height = 6
    fig= plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)

    plt.plot(epoch_num, train_metric, 'r',  label='Training Accuracy')
    plt.plot(epoch_num, val_metric, 'b',  label='Validation Accuracy')
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Metric-accuracy')
    plt.grid()
    plt.show()
    
    width = 8
    height = 6
    fig= plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    plt.plot(epoch_num, train_loss, 'r',  label='Training loss')
    plt.plot(epoch_num, val_loss, 'b',  label='Validation loss')
    ax.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Learning loss')
    plt.grid()
    plt.show()

    


from keras.utils import to_categorical
CLASS_7=['downstair','upstair','run','jump','walk', 'handwashing', 'exercise']
def one_hot_label(y_data, sub_video_dirs=CLASS_7):
    Mapping=dict()
    categories=len(sub_video_dirs)

    def hot_encode(target):
        label_y=np.zeros((1,categories))
        label_y[0][target]=1
        return label_y

    count=0
    for i in sub_video_dirs:
        Mapping[i]=count
        count=count+1

    y_features2=[]
    for i in range(len(y_data)):
        Type=y_data[i]
        lab=Mapping[Type]
        y_features2.append(lab)

    y_features=np.array(y_features2)
    y_features=y_features.reshape(y_features.shape[0],1)
    y_features = to_categorical(y_features)
    return y_features

def shuffle_data(data_x, data_y, data_z, data_w):
    sample_num = data_y.shape[0]
    
    index = np.arange(sample_num)
    np.random.shuffle(index)
    
    shuffle_x = data_x[index]
    shuffle_y = data_y[index]
    shuffle_z = data_z[index]
    shuffle_w = data_w[index]
    return shuffle_x, shuffle_y, shuffle_z, shuffle_w



def limit_data_learning(NEW_NET, ORG_NET, high_layers, New_train_limit,
                        Train_label_limit, New_Test_data, Test_label):
    time_start = time.time()
    print('\nNumber of available data: ',Train_label_limit.shape[0])
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    ############################### transferred model ####################
    personal_student = load_model('model/'+ORG_NET+'2'+NEW_NET+'_1_pre.hdf5')
    for layer in personal_student.layers:
         layer.trainable = False
    if high_layers == 'one':
        personal_student.add( Activation('relu', name = 'ACT_1'))
        personal_student.add( Dense(Train_label_limit.shape[1], activation='softmax', name = 'SOFT')  )
    elif high_layers == 'two':
        personal_student.add( Activation('relu', name = 'ACT_2'))
        personal_student.add( Dense(128, activation='relu', name = 'DENSE_ACT_1')  )
    #     model.add(Dropout(0.5))?
        personal_student.add( Dense(Train_label_limit.shape[1], activation='softmax', name = 'SOFT')  )

    epochs = 100
    batch_size = 128
    personal_student.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=sgd,
                  optimizer= adam,
                  metrics=['accuracy'])
    
    #rand_index = time.time()%100//1
    rand_index =np.random.rand()
    file_name = 'model/temp/temp_'+str(rand_index)+NEW_NET+'2'+ORG_NET+'.hdf5'
    checkpointer = ModelCheckpoint(filepath=file_name,
                                   verbose=0,
                                   monitor = 'val_acc',
                                   save_best_only=True)

#     print('Model saved at: '+file_name)
    H_personal = personal_student.fit( New_train_limit, Train_label_limit,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=0,
                                        shuffle=True,
                                        callbacks=[checkpointer], 
                                        validation_data=(New_Test_data, Test_label))
    plot_learning_curves(H_personal, no_fig = True)

    personal_student = load_model(file_name)
#     evaluate_model(personal_student, New_Test_data, Test_label)
    y_pred = np.argmax(personal_student.predict(New_Test_data), axis=1)
    y_true = np.argmax(Test_label, axis=1)
    accuracy_personal = sum(y_pred==y_true)/y_true.shape[0]
    print('Personalized model acc: ',accuracy_personal,'%.')
    
    ############################### scratch model ####################
    reg_value = 0.1

    if NEW_NET == 'sound':
        scratch_new_model = full_sound_model(New_train_limit, 
                                             Train_label_limit)
        scratch_new_model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=sgd,
                  optimizer= adam,
                  metrics=['accuracy'])
    elif NEW_NET == 'imu':
        scratch_new_model = full_imu_model(New_train_limit, 
                                           Train_label_limit)
        scratch_new_model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=sgd,
                  optimizer= adam,
                  metrics=['accuracy'])
    elif NEW_NET == 'video':
        scratch_new_model = full_video_model(New_train_limit, 
                                             Train_label_limit)
        scratch_new_model.compile(loss=keras.losses.categorical_crossentropy,
                                  optimizer=sgd,
                                  #optimizer= adam,
                                  metrics=['accuracy'])
    else:
        print("wrong new net!")
    
    
    epochs = 500
    if NEW_NET == 'video':
        epochs = 100
        
    batch_size = 128

    #scratch_new_model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=sgd,
    #              optimizer= adam,
    #              metrics=['accuracy'])
    file_name = file_name
    checkpointer = ModelCheckpoint(filepath=file_name,
                                   verbose=0,
                                   monitor = 'val_acc',
                                   save_best_only=True)

#     print('Model saved at: '+file_name)
    H_scratch = scratch_new_model.fit( New_train_limit, Train_label_limit,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=0,
                                        shuffle=True,
                                        callbacks=[checkpointer], 
                                        validation_data=(New_Test_data, Test_label))
    if NEW_NET == 'imu':
        plot_learning_curves(H_scratch)
    
    scratch_new_model = load_model(file_name)
    y_pred = np.argmax(scratch_new_model.predict(New_Test_data), axis=1)
    y_true = np.argmax(Test_label, axis=1)
    accuracy_scratch = sum(y_pred==y_true)/y_true.shape[0]
    print('Scratch model acc: ',accuracy_scratch,'%.')
    print('Time: ',time.time()-time_start)
    return accuracy_personal, accuracy_scratch
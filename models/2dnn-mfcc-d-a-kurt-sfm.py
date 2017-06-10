import numpy as np
import htkmfc as htk
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
# from keras.layers.convolutional import Convolution2D
# from keras.layers.convolutional import MaxPooling2D

import ConfigParser
import logging
import time
import sys
import os

np.random.seed(1337)
epoch=10 #Number of iterations to be run on the model while training
trainfile='/home/siddharthm/scd/combined/train-mfcc-kurt-sfm-labels.htk'
testfile='/home/siddharthm/scd/combined/test-mfcc-kurt-sfm-labels.htk'
valfile='/home/siddharthm/scd/combined/val-mfcc-kurt-sfm-labels.htk'
perc=0.5 #To control how much single speaker data we are letting in
def load_data_train(trainfile):
        print "Getting the overlap training data"
        a=htk.open(trainfile)
        train_data=a.getall()
        print "Done with Loading the training data: ",train_data.shape
        x_train=train_data[:,:-2] #Set to different column based on differrent model
        y_train=train_data[:,-1]
        del data
        return x_train,y_train
def load_data_test(testfile):
        a=htk.open(testfile)
        data=a.getall()
        print "Done loading the testing data: ",data.shape
        x_test=data[:,:-2]
        y_test=data[:,-1]
        del data
        return x_test,y_test
def load_data_val(valfile):
        a=htk.open(valfile)
        data=a.getall()
        print "Done loading the validation data: ",data.shape
        x_val=data[:,:-2]
        y_val=data[:,-1]
        return x_val,y_val

### THE MODEL and ALL ###
def seq(x_train,y_train,x_val,y_val,x_test,y_test):
        #Defining the structure of the neural network
        #Creating a Network, with 2 hidden layers.
        model=Sequential()
        model.add(Dense(256,activation='relu',input_dim=(1))) #Hidden layer1
        model.add(Dense(256,activation='relu')) #Hidden layer 2
        model.add(Dense(1,activation='sigmoid')) #Output Layer
        #Compilation region: Define optimizer, cost function, and the metric?
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        #Fitting region:Get to fit the model, with training data
        checkpointer=ModelCheckpoint(filepath=direc+common-save+'.json',monitor='val_acc',save_best_only=True,save_weights_only=True)

        #Doing the training[fitting]
        model.fit(x_train,y_train,nb_epoch=10,batch_size=batch,validation_data=(x_val,y_val),callbacks=[checkpointer])
        model.save_weights(direc+common-save+'-weights'+'.json') #Saving the weights from the model
        model.save(direc+common-save+'-model'+'.json')#Saving the model as is in its state

        ### SAVING THE VALIDATION DATA ###
        scores=model.predict(x_val,batch_size=batch)
        sio.savemat(direc+name_val+'.mat',{'scores':scores,'ytest':y_val}) #These are the validation scores.
        ### ------------- ###

        ### SAVING THE TESTING DATA ###
        scores_test=model.predict(x_test,batch_size=batch)
        sio.savemat(direc+name_test+'.mat',{'scores':scores_test,'ytest':y_test})
        ### ------------- ###

        ### For finding the details of classification ###
        #correct_overlap=0
        """for i in range(len(x_test)):
                if y_test[i]==1 and predictions[i]==1:
                        correct_overlap+=1
        print correct_overlap"""
        ### ------------- ###

#Non-function section
x_train,y_train=load_data_train(trainfile)
print "Loading training data complete"
x_test,y_test=load_data_test(testfile)
print "Loading testing data complete"
x_val,y_val=load_data_val(valfile)
print "Loading validation data complete"
# print "Shape test: ",x_train.shape," ",y_train.shape
#x_test,y_test=load_data_test(testfile)

#Some parameters for training the model
epoch=10 #Number of iterations to be run on the model while training
batch=1024 #Batch size to be used while training
direc="/home/siddharthm/scd/scores/"
common-save='2dnn-mfcc-d-a-kurt-sfm'
name_val=common-save+'-val'
name_test=common-save+'-test'
seq(x_train,y_train,x_val,y_val,x_test,y_test) #Calling the seq model, with 2 hidden layers
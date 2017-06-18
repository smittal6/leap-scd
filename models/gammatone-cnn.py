# coding: utf-8
# In[7]:

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
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

import ConfigParser
import logging
import time
import sys
import os

np.random.seed(1337)
epoch=10 #Number of iterations to be run on the model while training
trainfile='/home/siddharthm/scd/combined/gamma-labels-gender-train.htk'
testfile='/home/siddharthm/scd/combined/gamma-labels-gender-test.htk'
valfile='/home/siddharthm/scd/combined/gamma-labels-gender-val.htk'

#Now the data has the format that last column has the label, and the rest of stuff needs to be reshaped.
#The format for reshaping is as follows: Rows = Number of filters X Context size(40 in this case)
def cnn_reshaper(Data):
        dat=np.reshape(Data,(Data.shape[0],1,64,40)) #The format is: Number of samples, Channels, Rows, Columns
        return dat

def load_data_train(trainfile):
        print "Getting the training data"
        a=htk.open(trainfile)
        train_data=a.getall()
        print "Done with Loading the training data: ",train_data.shape
        data=train_data
        x_train=cnn_reshaper(data[:,:-1]) #Set to different column based on different model
        Y_train=data[:,-1]
        print Y_train.shape
        # print np.where(Y_train==2)
        Y_train=Y_train.reshape(Y_train.shape[0],1)
        y_train=np_utils.to_categorical(Y_train,2)
        del data
        return x_train,y_train
def load_data_test(testfile):
        a=htk.open(testfile)
        data=a.getall()
        print "Done loading the testing data: ",data.shape
        x_test=cnn_reshaper(data[:,:-1])
        Y_test=data[:,-1]
        print np.where(Y_test==2)
        # Y_test=np.reshape(Y_test,(Y_test.shape[0],1))
        # y_test=np_utils.to_categorical(Y_test,2)
        del data
        return x_test,Y_test
def load_data_val(valfile):
        a=htk.open(valfile)
        data=a.getall()
        print "Done loading the validation data: ",data.shape
        x_val=cnn_reshaper(data[:,:-1])
        Y_val=data[:,-1]
        Y_val=np.reshape(Y_val,(Y_val.shape[0],1))
        y_val=np_utils.to_categorical(Y_val,2)
        del data
        return x_val,y_val

def metrics(y_test,predictions,classes):
        #We have to modify this to include metrics to capture variations between male and female and blah-blah
        correct_change=0
        #print predictions[0:15,1]
        #classes=np_utils.probas_to_classes(predictions)
        print classes.shape
        print np.where(classes==1)
        for i in range(len(x_test)):
            if y_test[i]==1 and classes[i]==1:
                correct_change+=1
        print "Correct changes detected: ",correct_change
        print "Total speaker change frames: ",len(y_test[y_test==1])
        ### ------------- ###


### THE MODEL and ALL ###
def seq(x_train,y_train,x_val,y_val,x_test,y_test):
        #Defining the structure of the neural network
        #Creating a Network, with 2 Convolutional layers
        model=Sequential()
        model.add(Conv2D(64,(5,3)),activation='relu',input_shape=(1,64,40))
        model.add(Conv2D(64,(5,3),activation='relu'))
        model.add(MaxPooling2D((3,3)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(256,activation='relu')) #Fully connected layer 1
        model.add(Dropout(0.5))
        model.add(Dense(256,activation='relu')) #Fully connected layer 2
        model.add(Dropout(0.5))
        model.add(Dense(2,activation='softmax')) #Output Layer
        model.summary()
        #Compilation region: Define optimizer, cost function, and the metric?
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        #Fitting region:Get to fit the model, with training data
        checkpointer=ModelCheckpoint(filepath=direc+common_save+'.json',monitor='val_acc',save_best_only=True,save_weights_only=True)

        #Doing the training[fitting]
        model.fit(x_train,y_train,epochs=10,batch_size=batch,validation_data=(x_val,y_val),callbacks=[checkpointer])
        model.save_weights(direc+common_save+'-weights'+'.json') #Saving the weights from the model
        model.save(direc+common_save+'-model'+'.json')#Saving the model as is in its state

        ### SAVING THE VALIDATION DATA ###
        scores=model.predict(x_val,batch_size=batch)
        sio.savemat(direc+name_val+'.mat',{'scores':scores,'ytest':y_val}) #These are the validation scores.
        ### ------------- ###

        ### SAVING THE TESTING DATA ###
        scores_test=model.predict(x_test,batch_size=batch)
        sio.savemat(direc+name_test+'.mat',{'scores':scores_test,'ytest':y_test})
        ### ------------- ###
        # print model.evaluate(x_test,y_test,batch_size=batch)
        ### For finding the details of classification ###
        correct_change=0
        predictions=model.predict(x_test,batch_size=batch)
        classes=model.predict_classes(x_test,batch_size=batch)
        print "Shape of predictions: ", predictions.shape
        print "Shape of y_test: ",y_test.shape
        return y_test,predictions,classes

#Non-function section
x_train,y_train=load_data_train(trainfile)
print "Loading training data complete"
x_test,y_test=load_data_test(testfile)
print "Loading testing data complete"
x_val,y_val=load_data_val(valfile)
print "Loading validation data complete"

#Reahaping the data section 

#### ----- ####

### SHAPE TESTS ###
print "Train Shape: ",x_train.shape," ",y_train.shape
print "Test Shape: ",x_test.shape," ",y_test.shape
print "Val Shape: ",x_val.shape," ",y_val.shape
###

#Some parameters for training the model
epoch=20 #Number of iterations to be run on the model while training
batch=1024 #Batch size to be used while training
direc="/home/siddharthm/scd/scores/"
common_save='2dnn-mfcc-d-a-kurt-sfm'
name_val=common_save+'-val'
name_test=common_save+'-test'
y_test,predictions,classes=seq(x_train,y_train,x_val,y_val,x_test,y_test) #Calling the seq model, with 2 hidden layers


# In[8]:


metrics(y_test,predictions,classes)


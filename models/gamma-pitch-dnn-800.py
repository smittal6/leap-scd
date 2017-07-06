import numpy as np
import htkmfc as htk
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils,plot_model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import ConfigParser
import logging
import time
import sys
import os

np.random.seed(1337)
EPOCH=30 #Number of iterations to be run on the model while training

### TRAINFILE SECTION ###
trainfile='/home/siddharthm/scd/combined/gamma_pitch/800-gamma-pitch-labels-gender-train.htk'
### TESTFILE SECTION ###
testfile='/home/siddharthm/scd/combined/gamma_pitch/800-gamma-pitch-labels-gender-test.htk'
### VALIDATION FILE SECTION ###
valfile='/home/siddharthm/scd/combined/gamma_pitch/800-gamma-pitch-labels-gender-val.htk'

#Some parameters for training the model
batch=128 #Batch size to be used while training
direc="/home/siddharthm/scd/scores/"
common_save='800-gamma-pitch'
name_val=common_save+'-val'
#name_test=common_save+'-test'

def filter_data_train(x):
        stack1=x[x[:,-2]==1]
        np.random.shuffle(stack1)
        stack1=stack1[0:int(0.50*x.shape[0])]
        stack2=x[x[:,-2]==0]
        mat=np.vstack((stack1,stack2))
        np.random.shuffle(mat)
        return mat
def filter_data_val(x):
        stack1=x[x[:,-2]==1]
        np.random.shuffle(stack1)
        stack1=stack1[0:int(0.50*x.shape[0])]
        stack2=x[x[:,-2]==0]
        mat=np.vstack((stack1,stack2))
        np.random.shuffle(mat)
        return mat
#Now the data has the format that last column has the label, and the rest of stuff needs to be reshaped.
#The format for reshaping is as follows: Rows = Number of filters X Context size(40 in this case)
def cnn_reshaper(Data):
        dat=np.reshape(Data,(Data.shape[0],1,40,20)) #The format is: Number of samples, Channels, Rows, Columns
        return dat

def load_data_train(trainfile):
        print "Getting the training data"
        a=htk.open(trainfile)
        train_data=a.getall()
        print "Done with Loading the training data: ",train_data.shape
        data=filter_data_train(train_data)
        # x_train=cnn_reshaper(data[:,:-2]) #Set to different column based on different model
        x_train=data[:,:-2] #Set to different column based on different model
        scaler=StandardScaler().fit(x_train)
        # x_train=scaler.transform(x_train)
        Y_train=data[:,-2]
        print Y_train.shape
        # print np.where(Y_train==2)
        Y_train=Y_train.reshape(Y_train.shape[0],1)
        y_train=np_utils.to_categorical(Y_train,2)
        print y_train[0:5,:]
        gender_train=data[:,-1]
        del data
        #x_train has complete data, that is gammatone and also the pitch variance values.
        return x_train,y_train,gender_train,scaler

def load_data_test(testfile):
        a=htk.open(testfile)
        data=a.getall()
        print "Done loading the testing data: ",data.shape
        x_test=cnn_reshaper(data[:,:-2])
        Y_test=data[:,-2]
        print np.where(Y_test==2)
        # Y_test=np.reshape(Y_test,(Y_test.shape[0],1))
        # y_test=np_utils.to_categorical(Y_test,2)
        gender_labels=data[:,-1]
        del data
        return x_test,Y_test,gender_labels
def load_data_val(valfile,scaler):
        a=htk.open(valfile)
        data=a.getall()
        print "Done loading the validation data: ",data.shape
        data=filter_data_val(data)
        x_val=data[:,:-2]
        # x_val=scaler.transform(x_val)
        Y_val=data[:,-2]
        # print np.where(Y_val==1)
        Y_val=np.reshape(Y_val,(Y_val.shape[0],1))
        y_val=np_utils.to_categorical(Y_val,2)
        # print np.where(y_val[:,1]==1)
        gender_val=data[:,-1]
        del data
        #x_val has the pitch variances and also the gammatone values
        return x_val,y_val,gender_val

def data_saver(data):
        os.chdir('/home/siddharthm/scd/scores')
        f=open(common_save+'-complete.txt','a')
        f.write('\n')
        f.write(str(data))
        f.close()

def metrics(y_val,classes,gender_val):
        #We have to modify this to include metrics to capture variations between male and female and blah-blah
        #initializing the two matrixes to be saved.
        cd_correct_matrix=np.zeros((2,2))
        single_correct_matrix=np.zeros((1,2))
        cd_incorrect_matrix=np.zeros((2,2))
        single_incorrect_matrix=np.zeros((1,2))
        # print classes.shape
        single_correct,cd_correct,single_incorrect,cd_incorrect=0,0,0,0
        print "Predicted Classes: ",np.where(classes==1)
        print "Actual classes: ",np.where(y_val[:,1]==1)
        for i in range(len(y_val)):
                if y_val[i,1]==1:
                        if classes[i]==1:
                                cd_correct+=1
                        else:
                                cd_incorrect+=1
                elif y_val[i,0]==1:
                        if classes[i]==0:
                                single_correct+=1
                        else:
                                single_incorrect+=1

        data_saver('Speaker changes Correct detected')
        data_saver(cd_correct)
        data_saver('Speaker changes wrongly classified')
        data_saver(cd_incorrect)
        data_saver('Single speaker frames correct')
        data_saver(single_correct)
        data_saver('Single speaker frames wrongly classified')
        data_saver(single_incorrect)
        print "Gender Labels[0:100]: ",gender_val[0:100]

        #We need a matrix, one of correctly classified changes, and the other of incorrectly classified changes.
        for i in range(len(y_val)):
                if y_val[i,1]==1:
                        id1=int(str(gender_val[i])[0])-1
                        id2=int(str(gender_val[i])[1])-1
                        if classes[i]==1:
                                cd_correct_matrix[id1,id2]+=1
                        else:
                                cd_incorrect_matrix[id1,id2]+=1
                elif y_val[i,0]==1:
                        gid=int(gender_val[i]-1) #1 female, 2 male.
                        # print "Single Gender Id: ",gid
                        if classes[i]==0:
                                single_correct_matrix[0,gid]+=1
                        else:
                                single_incorrect_matrix[0,gid]+=1
        data_saver('Speaker changes Correct detected')
        data_saver(cd_correct_matrix)
        data_saver('Speaker changes wrongly classified')
        data_saver(cd_incorrect_matrix)
        data_saver('Single speaker frames correct')
        data_saver(single_correct_matrix)
        data_saver('Single speaker frames wrongly classified')
        data_saver(single_incorrect_matrix)
        ### ------------- ###

#Load training data
x_train,y_train,gender_train,scaler=load_data_train(trainfile)
print "Loading training data complete"
# px_train,py_train,fgender_train,scaler=load_data_train(pitch_trainfile)
# print "Loading pitch training data complete"

##Load test data
#x_test,y_test,gender_labels=load_data_test(testfile)
#print "Loading testing data complete"

##Load validation data
x_val,y_val,gender_val=load_data_val(valfile,scaler)
print "Loading validation data complete"
# px_val,py_val,pgender_val=load_data_val(fbank_valfile,scaler)
# print "Loading pitch validation data complete"

## SHAPE TESTS ###
print "Train Shape: ",x_train.shape," ",y_train.shape
#print "Test Shape: ",x_test.shape," ",y_test.shape
print "Val Shape: ",x_val.shape," ",y_val.shape
###

### THE MODEL and ALL ###
def seq(x_train,y_train,x_val,y_val,x_test,y_test):
        #Defining the structure of the neural network

        # Creating the first model, which takes as input the gammatone values
        model1=Sequential()
        model1.add(Dense(1024,activation='relu',input_shape=(5184,)))
        model1.add(Dense(1024,activation='relu'))
        model1.add(Dropout(0.25))
        model1.add(Dense(512,activation='relu'))
        model1.add(Dropout(0.25))
        model1.add(Dense(256,activation='relu'))
        model1.add(Dense(128,activation='relu'))
        #Creating the second model, which takes pitch variance as input
        # model2=Sequential()
        # model2.add(Input(shape=(1,)))

        a2 = Input(shape =(1,)) #creating the input
        # f2 = model2(a2) #making the model

        a1 = Input(shape=(5184,)) #Just creating the input acceptance for gammatone network
        f1 = model1(a1) #make the model

        y = concatenate([f1, a2]) #concatenating the output of two models.
        y = Dense(64,activation='relu')(y)
        x = Dense(2, activation='softmax')(y) #Linking the model to the output

        model = Model(inputs=[a1, a2], outputs=x) #calling the combined model
        plot_model(model,to_file='gp-800.jpg')
        model.summary()
        ### SAVE MODEL STUFF ###
        data_saver("##### -------- #####")
        # data_saver(str(model.to_json()))

        ### Some crucial calls ###
        sgd=SGD(lr=1)
        early_stopping=EarlyStopping(monitor='val_loss',patience=6)
        reduce_lr=ReduceLROnPlateau(monitor='val_loss',patience=6,factor=0.5,min_lr=0.0000001)
        checkpointer=ModelCheckpoint(filepath=direc+common_save+'.json',monitor='val_acc',save_best_only=True,save_weights_only=True)

        #Compilation region: Define optimizer, cost function, and the metric?
        model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

        #Fitting region: Get to fit the model, with training data

        #Doing the training[fitting]
        model.fit([x_train[:,0:-1],x_train[:,-1]],y_train,epochs=EPOCH,batch_size=batch,validation_data=([x_val[:,0:-1],x_val[:,-1]],y_val),callbacks=[checkpointer,early_stopping,reduce_lr])
        model.save_weights(direc+common_save+'-weights'+'.json') #Saving the weights from the model
        model.save(direc+common_save+'-model'+'.json')#Saving the model as is in its state

        ### SAVING THE VALIDATION DATA ###
        scores=model.predict([x_val[:,0:-1],x_val[:,-1]],batch_size=batch)
        sio.savemat(direc+name_val+'.mat',{'scores':scores,'ytest':y_val}) #These are the validation scores.
        classes=scores.argmax(axis=-1)
        ### ------------- ###

        ### SAVING THE TESTING DATA ###
        #scores_test=model.predict(x_test,batch_size=batch)
        #sio.savemat(direc+name_test+'.mat',{'scores':scores_test,'ytest':y_test})
        ### ------------- ###
        # print model.evaluate(x_test,y_test,batch_size=batch)

        #predictions=model.predict(x_val,batch_size=batch)
        #print "Shape of predictions: ", predictions.shape
        data_saver(str(len(np.where(y_train[:,0]==1)[0])))
        data_saver(str(len(np.where(y_train[:,1]==1)[0])))
        print "Training 0 class: ",len(np.where(y_train[:,0]==1)[0])
        print "Training 1 class: ",len(np.where(y_train[:,1]==1)[0])
        return classes

#Non-function section

#y_test,predictions,classes=seq(x_train,y_train,x_val,y_val,x_test,y_test) #Calling the seq model, with 2 hidden layers
classes=seq(x_train,y_train,x_val,y_val,0,0) #Calling the seq model, with 2 hidden layers
metrics(y_val,classes,gender_val)


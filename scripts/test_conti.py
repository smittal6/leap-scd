import keras
import numpy as np
import htkmfc as htk
import scipy.io as sio
import time
import re
from keras.models import Sequential,Model,load_model
from keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Input,Flatten,Activation,Merge,Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import load_model

frames=11 #Context frames.
np.random.seed(777)

def mean_var_finder(data):
        mean=np.mean(data,axis=0)
        var=np.var(data,axis=0)
        return (mean,var)

def normalizer(data,mean,var):
        data=(data-mean)/np.sqrt(var)
        return data

def cnn_reshaper(data):
        data=np.reshape(data,(data.shape[0],1,frames,-1))
        return data

def Data_Getter(filename):
        gamma=htk.open('/home/siddharthm/scd/context/600/gamma/train/'+filename+'.htk') #Getting gamma context feats
        pitch=htk.open('/home/siddharthm/scd/context/600/pitch/train/'+filename+'.htk') #Getting pitch context feats
        temp_gamma=gamma.getall()
        temp_pitch=pitch.getall()
        only_pitch=temp_pitch[:,0] #Extracting only the pitch value
        x_val=temp_gamma[:,:-1] #Only gammatone values, here 64*61
        y_val=temp_gamma[:,-1] #These are the real labels, that is from the ground truth
        y_val=y_val.reshape(y_val.shape[0],1)
        y_val=y_val.astype(np.int8)
        print(x_val.shape,y_val.shape)
        return (x_val,only_pitch,y_val)

###---- LIST FILE SECTION-----###
list_file=open('/home/siddharthm/scd/lists/rawtrainfiles.list')
List=list_file.read()
List=List.strip()
List=re.split('\n',List)
print "Total number of files in this list: ",len(List)
###-------###

###---- Load the saved model[architecture, along with the weights]-----###
model=load_model('600-gamma-pitch-model.h5')
###-------###

for i in range(len(List)):
        print "The file is: ",List[i] #printing which file we are considering.
        x_val,pitch,y_truth=Data_Getter(List[i]) #Gets the raw input, and the ground truth file
        Scores=model.predict(Context_val,batch_size=256)
        print(Scores.shape)
        sio.savemat('/home/neerajs/work/codes/results/discard_meeting/'+List[i]+'.mat',{'Scores':Scores,'Y_train':Y_train})

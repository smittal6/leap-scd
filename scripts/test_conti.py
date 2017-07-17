import keras
import numpy as np
import htkmfc as htk
import scipy.io as sio
import time
import re
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Input,Flatten,Activation,Merge,Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import load_model

frames=11
np.random.seed(2308)

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
        val=htk.open('/home/neerajs/work/codes/from_11/ami_fbank_all_val_discard/'+filename+'.htk')
        X_val=val.getall()
#	y=htk.open('/home/neerajs/work/ami/feats/context/11/discard/val/labels/'+filename+'.htk')
#	Y_val=y.getall()
        Y_val=X_val[:,-1]
        X_val=X_val[:,:-1]
        Y_val=Y_val.reshape(Y_val.shape[0],1)
        Y_val=Y_val.astype(np.int8)
        print(X_val.shape,Y_val.shape)
        return (X_val,Y_val)

List_file=open('/home/neerajs/work/codes/results/list.list')
List=List_file.read()
List=List.strip()
List=re.split('\n',List)
print(len(List))

nb_filters=256
filter_size_x=3
filter_size_y=7


model=Sequential()
model.add(Conv2D(64,(5,7),input_shape=(1,11,64)))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,5)))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.load_weights('best_model_timit_meeting.h5')

data=sio.loadmat('mean_var_timit_meeting.mat')
mean=data['mean']
var=data['var']


for i in range(len(List)):
	print(i)
	X_val,Y_train=Data_Getter(List[i])
	Context_val=cnn_reshaper(X_val)	
	Context_val=normalizer(Context_val,mean,var)
	Scores=model.predict(Context_val,batch_size=256)
	print(Scores.shape)
#	Scores=np.reshape(Scores,(Scores.shape[0],1))
	sio.savemat('/home/neerajs/work/codes/results/discard_meeting/'+List[i]+'.mat',{'Scores':Scores,'Y_train':Y_train})



import numpy as np
import htkmfc as htk
import scipy.io as sio
from multiprocessing import Process

import ConfigParser
import logging
import time
import sys
import os

np.random.seed(1337)
epoch=10 #Number of iterations to be run on the model while training
trainfile='/home/siddharthm/scd/combined/gamma-labels-train.htk'
testfile='/home/siddharthm/scd/combined/gamma-labels-test.htk'
valfile='/home/siddharthm/scd/combined/gamma-labels-val.htk'
filename_train='clean-gamma-labels-train'
filename_test='clean-gamma-labels-test'
filename_val='clean-gamma-labels-val'

def changedir():
        os.chdir('/home/siddharthm/scd/combined')
        print "Current working directory: ",os.getcwd()

def filter_data(x):
        ### Filter the data. That is only keep 0 or 1 classes.
        return x[ (x[:,-1]==0)|(x[:,-1]==1)]

def load_data_train(trainfile):
        a=htk.open(trainfile)
        train_data=a.getall()
        print "Done with Loading the training data: ",train_data.shape
        data=filter_data(train_data)
        print "Filtered train data shape: ",data.shape
        changedir()
        writer=htk.open(filename_train+'.htk',mode='w',veclen=data.shape[1])
        del data

def load_data_test(testfile):
        a=htk.open(testfile)
        data=a.getall()
        print "Done loading the testing data: ",data.shape
        data=filter_data(data)
        print "Filtered test data shape: ",data.shape
        changedir()
        writer=htk.open(filename_test+'.htk',mode='w',veclen=data.shape[1])
        del data

def load_data_val(valfile):
        a=htk.open(valfile)
        data=a.getall()
        print "Done loading the validation data: ",data.shape
        data=filter_data(data)
        print "Filtered test data shape: ",data.shape
        changedir()
        writer=htk.open(filename_val+'.htk',mode='w',veclen=data.shape[1])
        del data

load_data_val(valfile)
load_data_test(testfile)
load_data_train(trainfile)

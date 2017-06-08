import re
import os
import sys
import time
import numpy as np
import scipy.io as sio
import htkmfc as htk
def file_opener(file_read):
        file_reader=open(file_read)
        file_reader=file_reader.read()
        file_reader=file_reader.strip()
        file_reader=re.split('\n',file_reader)
        return file_reader

def data_creator(num,addr,file_reader,filename):
        corrupt_files=0
        ind=0
        matrix=np.empty((1,num))
        writer=htk.open(filename+'.htk',mode='w',veclen=num) #num is the final feature vector size to be written(including the label. Ensure that by looking at the botttom entry)
        for i in range(len(file_reader)):
                print "Starting with file: ",i
                data_read=htk.open(addr+file_reader[i]+'.htk') #opening the MFCC HTK file
                kurt_matrix=sio.loadmat(kurt_addr+file_reader[i]+'.mat')['kurt'] #opening the kurtosis matrix for a file
                sfm_matrix=sio.loadmat(sfm_addr+file_reader[i]+'.mat')['sfm'] #opening the sfm_matrix file
                labels_this_file=sio.loadmat(label_addr+file_reader[i]+'.mat')['labels']

                ### Kurtosis and sfm are row vectors, that is (1,Number of frames)
                ### MFCC_D_A -- KURT -- SFM -- LABEL   <--- Structure of the final matrix
                #try:
                read_data=data_read.getall()
                kurt_vector=np.transpose(kurt_matrix)
                sfm_vector=np.transpose(sfm_matrix)
                label_vector=np.transpose(labels_this_file)
                print "Got all the vectors"
                print read_data.shape,kurt_vector.shape,sfm_vector.shape,label_vector.shape
                final_vector=np.hstack((read_data,kurt_vector,sfm_vector,label_vector))
                print "Horizontally stacking done"
                matrix=np.vstack((matrix,final_vector))
                print "Vertical stacking done"
                del read_data
                        #except:
                                #       corrupt_files+=1
                                #      continue
                                #ind=ind+read_data.shape[0]

        writer.writeall(matrix)
        print('corrput_files',corrupt_files)
        #   	labels=np.ones((ind,1))
        #   	labels=labels*index
        #	print(labels.shape)
        #	wri=htk.open(filename+'labels.htk',mode='w',veclen=1)
        #	wri.writeall(labels)

if os.path.isdir('/home/siddharthm/scd/combined'):
        pass
else:
        os.chdir('/home/siddharthm/scd/combined')

addr='/home/siddharthm/scd/feats/mfcc/train/'#address of the HTK files stored somewhere
kurt_addr='/home/siddharthm/scd/feats/kurt/train/'
sfm_addr='/home/siddharthm/scd/feats/sfm/train/'
label_addr='/home/siddharthm/scd/vad/train/'
num=39+1+1+1 #The length of the feature vector, to be read and stored in the htk format[Right now, 39 MFCC_D_A+1 KURT+ 1 SFM+1 Label]
file_read='/home/siddharthm/scd/lists/rawtrainfiles.list' #The raw filenames, in the form of list
filename='mfcc-kurt-sfm-labels' #The name of the file where stuff is going to be stored
#file_reader=file_opener(file_read) #Calling the function to read the list of files
file_reader=['MZMB0_X86-MTJG0_X350-830']
data_creator(num,addr,file_reader,filename) #Finally call the data creator

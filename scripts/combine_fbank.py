import re
import os
import sys
import time
import numpy as np
import scipy.io as sio
import htkmfc as htk

# If female then mark her as 0 and if male then mark as 1
# Now for any kind of combination, we take both the ids and concatenate them, and store as int

percentage_to_keep=0.2
keep_true=1

def return_vec(x,id1,id2):
        vector=np.zeros((len(x),1))
        first_index=np.where(x==1)[0][0] #Storing the part from where speaker change and subsequently second speaker starts
        for i in range(len(x)):
                if x[i]==1:
                        vector[i,0]=int(str(id1)+str(id2))
                if x[i]==0:
                        if i<first_index:
                                vector[i,0]=id1
                        else:
                                vector[i,0]=id2
        if len(vector)==len(x):
                # print vector
                return vector
        else:
                print "Something is wrong in the return_vec function"

def filter_data(x):
        type1=x[x[:,-1]==0]
        type2=x[x[:,-1]==1]
        keep=int(percentage_to_keep*type1.shape[0])
        keep2=int(keep_true*type2.shape[0])
        np.random.shuffle(type2)
        np.random.shuffle(type1)
        type1=type1[0:keep,:]
        type2=type2[0:keep2,:]
        return_mat=np.vstack((type1,type2))
        return return_mat

def file_opener(file_read):
        file_reader=open(file_read)
        file_reader=file_reader.read()
        file_reader=file_reader.strip()
        file_reader=re.split('\n',file_reader)
        return file_reader
def changedir():
        os.chdir(cwd)
        print "Current working directory: ",os.getcwd()

def data_creator(num,addr,file_reader,filename):
        corrupt_files=0
        noscdlab=0
        scdlab=0
        matrix=np.empty((0,num))
        changedir()
        writer=htk.open(filename+'.htk',mode='w',veclen=num) #num is the final feature vector size to be written(including the label. Ensure that by looking at the botttom entry)
        for i in range(len(file_reader)):
                print "Starting with file: ",i
                data_read=htk.open(addr+file_reader[i]+'.htk') #opening the Gamma-Label HTK file
                # kurt_matrix=sio.loadmat(kurt_addr+file_reader[i]+'.mat')['kurt'] #opening the kurtosis matrix for a file
                # sfm_matrix=sio.loadmat(sfm_addr+file_reader[i]+'.mat')['sfm'] #opening the sfm_matrix file
                # labels_this_file=sio.loadmat(label_addr+file_reader[i]+'.mat')['labels']

                ### Kurtosis and sfm are row vectors, that is (1,Number of frames)
                ### GAMMATONE -- LABEL --GenderLabel  <--- Structure of the final matrix
                try:
                        read_data=data_read.getall()
                        id1=(1,2)[(file_reader[i][0]=='M')==True]
                        temp_index=file_reader[i].index("-")
                        id2=(1,2)[(file_reader[i][temp_index+1]=='M')==True]
                        gender_label=return_vec(read_data[:,-1],id1,id2)
                        read_data=np.hstack((read_data,gender_label))
                        read_data=filter_data(read_data)
                        scdlab+=len(np.where(read_data[:,-1]==1)[0])
                        noscdlab+=read_data.shape[0]-len(np.where(read_data[:,-1]==1)[0])
                        #id1 and id2 are integers essentially. if male then 1, if female than 0
                        # kurt_vector=np.transpose(kurt_matrix)
                        # sfm_vector=np.transpose(sfm_matrix)
                        # label_vector=np.transpose(labels_this_file)
                        # final_vector=np.hstack((read_data,kurt_vector,sfm_vector,label_vector))
                        final_vector=read_data
                        # matrix=np.vstack((matrix,final_vector))
                        del read_data
                except:
                        print "In the corrupt file section"
                        corrupt_files+=1
                        continue
                        # ind=ind+read_data.shape[0]
                #HTK supports concatenation, so we don't have to deal with numpy matrix again and again
                writer.writeall(final_vector)
        print('corrput_files',corrupt_files)
        f=open(save_extra,'w')
        write_string=str(scdlab)+","+str(noscdlab)+", Corrupt: "+str(corrupt_files)
        f.write(write_string)
        f.close()

# First sys input is whether test/, train/ or val/ and second input is trainfile.list or ...., third is train, test or val
addr='/home/siddharthm/scd/context/200/fbank/'+str(sys.argv[1])#address of the HTK files stored somewhere
cwd='/home/siddharthm/scd/combined/fbank' #The directory where we will change the address of 
# kurt_addr='/home/siddharthm/scd/feats/kurt/'+str(sys.argv[1])
# sfm_addr='/home/siddharthm/scd/feats/sfm/'+str(sys.argv[1])
# label_addr='/home/siddharthm/scd/vad/'+str(sys.argv[1])
num=20*40+1+1 #The length of the feature vector, to be read and stored in the htk format[Right now,20*40 Fbank +1 Label+1 gender]
file_read='/home/siddharthm/scd/lists/'+str(sys.argv[2]) #The raw filenames, in the form of list
filename='200-fbank-labels-gender-'+str(sys.argv[3]) #The name of the file where stuff is going to be stored
save_extra='200-fbank-extra-'+str(sys.argv[3])+'.txt' #For saving the count of lables
file_reader=file_opener(file_read) #Calling the function to read the list of files
# file_reader=['FAJW0_I1263-FCYL0_X349-9346','FAJW0_I1263-FGCS0_X226-13892']
data_creator(num,addr,file_reader,filename) #Finally call the data creator

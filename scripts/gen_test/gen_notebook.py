
#The idea is to just generate the wav file and the ground truth for speaker change detection.
import re
import numpy as np
import scipy.io.wavfile as wav
import scipy
import scipy.io as sio
import os
import htkmfc as htk
#### DO NOT CHANGE ####
# base='/home/neerajs/work/NEW_REGIME/WAV/'
base='/home/siddharthm/'
clean='clean_wav/'
rev='rev_wav/'
rev_noise='rev_noise_wav/'
rev_inaud='rev_inaud_wav/'
phndir='/home/siddharthm/TIMIT/phones/'
choices=[clean,rev,rev_noise,rev_inaud]
#### ------------- ####

wavesavdir='/home/siddharthm/scd/wav/test/' #Save the generated wave file in this dir
over_addr='/home/siddharthm/scd/vad/test/' #labels in the directory
common_save='test_combinations'
### SOME VARIABLE DEFINITIONS ###
ratio_sc=0.20
ratio_sil=0.70
time_decision=200 #in milliseconds
decision_samples=time_decision*16 #Assuming 16KHz sampling rate
silence_samples=0.10*decision_samples #The number of samples to be inserted[as silence]
###

# We are constructing ground truth from Phone files.
# There are many ways in which we can generate speaker change files. Right now, we are generating by concatenating the two files.

def data_saver(data):
        os.chdir('/home/siddharthm/scd/scores')
        f=open(common_save+'.list','a')
        f.write('\n')
        f.write(str(data))
        f.close()

def gen_func(file1,file2,input_index):
        print('Begin')
        print(file1,file2,input_index)
        data_saver(str(file1)+str(file2))
        ind1=file1.index('_')+1
        ind2=file2.index('_')+1
        file1_mod=file1[0:ind1]+file1[ind1+1:]
        file2_mod=file2[:ind2]+file2[ind2+1:]
        #Fetching the base directory of the working files
        wav_addr1=base+choices[np.random.randint(1)] #Keeping only the clean files
        wav_addr2=base+choices[np.random.randint(1)] #Same as above
        #reading the two wav files and overlapping them
        wav_file1_mod=wav_addr1+file1_mod+'.wav'
        wav_file1=wav_addr1+file1+'.wav'
        wav_file2_mod=wav_addr2+file2_mod+'.wav'
        wav_file2=wav_addr1+file2+'.wav'
        # print(wav_file2)
        phnFile1=phndir+file1+'.PHN'
        phnFile2=phndir+file2+'.PHN'

        #### Check if the first line begins with 0 for h#

        nsFile1=[]
        phnFile1Fid=open(phnFile1).readlines()
        if int(phnFile1Fid[0].split(' ')[0]) != 0:
                nsFile1.append([0,int( phnFile1Fid[0].split(' ')[1])/160])
        for line in phnFile1Fid:
                line=line.rstrip()
                if re.search(('h#|epi|sil'),line):
                        nsFile1.append([int(line.split(' ')[0])/160,int(line.split(' ')[1])/160])
        # print "Silence regions, file1: "
        # print nsFile1
        nsFile2=[]
        phnFile2Fid=open(phnFile2).readlines()
        if int(phnFile2Fid[0].split(' ')[0]) != 0:
                nsFile2.append([0,int( phnFile2Fid[0].split(' ')[0])/160])

        for line in phnFile2Fid:
                line=line.rstrip()
                if re.search(('h#|epi|sil'),line):
                        nsFile2.append([int(line.split(' ')[0])/160,int(line.split(' ')[1])/160])
        # print "Silence regions, File 2: "
        # print nsFile2
        #Wavfile.read returns the sampling rate and the read data. The sampling rate is assumed to be 16KHz for our purposes.
        try:
                [a1,a2]=wav.read(wav_file1)
                a2=np.reshape(a2,(1,a2.shape[0])) #a2 is the actual data sample, reshaping it to (1,size)
        except IOError:
                [a1,a2]=wav.read(wav_file1_mod)
                a2=np.reshape(a2,(1,a2.shape[0])) #a2 is the actual data sample, reshaping it to (1,size)

        try:
                [b1,b2]=wav.read(wav_file2)
                b2=np.reshape(b2,(1,b2.shape[0])) #b2 is the sample, and reshaping it
        except IOError:
                [b1,b2]=wav.read(wav_file2_mod)
                b2=np.reshape(b2,(1,b2.shape[0])) #b2 is the sample, and reshaping it

        #Making them floats
        a2=a2.astype(float)
        b2=b2.astype(float)

        #### FRAME LEVEL MANIPULATIONS FOR CREATING OVERLAP LABELS ####
        nFrames1=int(a2.shape[1]/160) #Number of frames that are possible from File one
        nFrames2=int(b2.shape[1]/160) #Number of frames from File 2

        # print nFrames1,nFrames2,overlap_frame,nsFile1,nsFile2
        stFrame2=nFrames1
        totalFrame=nFrames1+nFrames2-1
        # print stFrame2,totalFrame,nsFile2[0][0]+stFrame2 

        labelFrames1=np.ones((totalFrame,), dtype=np.int)
        for i in range(len(nsFile1)):
                labelFrames1[nsFile1[i][0]:nsFile1[i][1]] = 0
        labelFrames1[nFrames1:]=0

        labelFrames2=np.full((totalFrame,), 2, dtype=np.int) #Giving different identity to second speaker file[By using 2]
        labelFrames2[:stFrame2]=0
        for i in range(len(nsFile2)):
                labelFrames2[nsFile2[i][0]+stFrame2:nsFile2[i][1]+stFrame2] = 0
        ### GENERATING THE FINAL LABEL FILE ###
        # print type(labelFrames1)
        lastindex=np.where(labelFrames1==1)[0][-1]
        firstindex=np.where(labelFrames2==2)[0][0]
        silence_part=np.zeros((int(silence_samples/160),),dtype=np.int)
        silence_actual_wav=np.zeros((int(silence_samples),))
        # print "Number of frames in File 1",nFrames1
        # print "Last non zero File1: ",lastindex,",First File 2: ",firstindex,",Length of complete vector: ",labelFrames2.shape
        labelpart1=np.hstack((labelFrames1[0:lastindex+1],silence_part))
        labelpart2=np.hstack((labelFrames2[firstindex:])) #silence part was not extraneous, was coming twice
        labelFrames = np.hstack((labelpart1,labelpart2))
        # print "Length of the labelFRames vector: ",labelFrames.shape
        start=0
        iterator=0
        skip_entries=int(decision_samples/160)
        # print "Skip entries", skip_entries #By skip entries we mean the entries to be skipped in the label vector
        end=start+skip_entries
        ### GENERATING THE ACTUAL WAVE FILE ###
        # print "Actual samples from silence region: ",silence_actual_wav.shape[0]
        # print "Samples from First file: ",160*(lastindex+1)
        samplestart=160*labelFrames.shape[0]-(silence_actual_wav.shape[0]+160*(lastindex+1))
        out=np.hstack((a2[0,:160*(lastindex+1)],silence_actual_wav,b2[0,-samplestart:-1])) #Actually creating the numpy array which has the overlap and single speaker speech segments
        # print "Out wav file: ", out.shape
        flabels=[]
        count=0
        flag=0
        # 2 for the silence class, 1 for speaker change frame, 0 for no speaker change frame
        while end<len(labelFrames):
                #Getting the vector ready
                aconsider=labelFrames[start:end]
                # print "Length of samples under consideration: ",160*len(aconsider)
                #Some definitions for further calculations
                count_zero=len(np.where( aconsider == 0 )[0])
                count_one=len(np.where( aconsider == 1 )[0])
                count_two=len(np.where( aconsider == 2 )[0])
                #Decision section
                if count_zero*160>int(ratio_sil*decision_samples):
                        dec=2
                elif min(count_one,count_two)*160>int(ratio_sc*decision_samples):
                        dec=1
                else:
                        dec=0
                #Update section
                # print "Decision Taken: ",dec
                # print count_zero,count_one,count_two,dec
                if dec==2:
                        count_two+=1
                if dec==1:
                        count+=1
                else:
                        flag+=1
                flabels.append(dec)
                # print "Flag: ", flag,"Count: ",count
                iterator+=1
                start+=1
                end=skip_entries+start

        print count,flag,count_two
        # print iterator
        ### Setting the labels and the output ###
        out=out.astype(np.int16)
        out=np.reshape(out,(out.shape[0],1)) #Reshaping it to form the vector which is required to be written in wav file

        # print "The shape of out vector: ",out.shape
        flabels=np.array(flabels)
        # print flabels.shape
        # print type(flabels)
        flabels=np.reshape(flabels,(1,flabels.shape[0]))
        # print flabels.shape
        ### SAVING THE STUFF SECTION ###
        scipy.io.wavfile.write(wavesavdir+file1+'-'+file2+'-'+str(input_index)+'.wav',a1,out)
        writer=htk.open(over_addr+file1+'-'+file2+'-'+str(input_index)+'.htk',mode='w',veclen=max(flabels.shape))
        writer.writeall(flabels)
        ### --------- ###



### TRY CALLS[Actual use with wrapper] ###
# gen_func('MPGR0_I1410','MTPF0_I1865',1)
# gen_func('FTBW0_X85','FTLG0_I1743',2)
### ------- ###


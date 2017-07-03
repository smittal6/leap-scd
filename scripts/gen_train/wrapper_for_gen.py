import re
import numpy as np
import gen_notebook as gen

f=open('/home/siddharthm/scd/vad/trainfile1.list')
f=f.read()
f=f.strip()
f=re.split('\n',f)

f_1=open('/home/siddharthm/scd/vad/trainfile2.list')
f_1=f_1.read()
f_1=f_1.strip()
f_1=re.split('\n',f_1)

f_2=open('/home/siddharthm/scd/vad/trainnumbers.list')
f_2=f_2.read()
f_2=f_2.strip()
f_2=re.split('\n',f_2)

for i in range(len(f)-10000):
        # [index1,index2]=np.random.choice(3200,size=2,replace=False)
        file1=f[i]
        file2=f_1[i]
        gen.gen_func(file1,file2,int(f_2[i]))
        print "Completed Generating",  i+1


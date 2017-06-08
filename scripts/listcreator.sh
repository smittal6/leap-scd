
#First create the raw.wav files
ls ../WAV/train/ $search_path > rawtrainwav.list
#Then create the just raw names, would have to use sed
cp rawtrainwav.list rawtrainfiles.list
sed -i "s|\..*||g" rawtrainfiles.list

cp rawtrainwav.list rawtrainaddrhtk.list
cp rawtrainwav.list rawtrainaddrwav.list
#addrhtk will contain the addr of htk files, addr wav will contain the addr of wav files[along with the files itself]
sed -i "s|^|/home/neerajs/work/NEW_REGIME/SID/FEATS/mfcc/train/|g" rawtrainaddrhtk.list
sed -i "s|^|/home/neerajs/work/NEW_REGIME/SID/WAV/train/|g" rawtrainaddrwav.list

#Now let us make the final list required for HTK generation, which requires first wav file and then htk file
sed -i "s|wav|htk|g" rawtrainaddrhtk.list #replacing the wav extension by htk extension
paste rawtrainaddrwav.list rawtrainaddrhtk.list > htkfinaltrain.list

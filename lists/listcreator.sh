direc="/home/siddharthm/scd/wav/val"
#First create the raw.wav files
ls $direc $search_path > rawvalwav.list
#Then create the just raw names, would have to use sed
cp rawvalwav.list rawvalfiles.list
sed -i "s|\..*||g" rawvalfiles.list

cp rawvalwav.list rawvaladdrhtk.list
cp rawvalwav.list rawvaladdrwav.list
#addrhtk will contain the addr of htk files, addr wav will contain the addr of wav files[along with the files itself]
sed -i "s|^|/home/siddharthm/scd/feats/mfcc/val/|g" rawvaladdrhtk.list
sed -i "s|^|/home/siddharthm/scd/wav/val/|g" rawvaladdrwav.list

#Now let us make the final list required for HTK generation, which requires first wav file and then htk file
sed -i "s|wav|htk|g" rawvaladdrhtk.list #replacing the wav extension by htk extension
paste rawvaladdrwav.list rawvaladdrhtk.list > htkfinalval.list
rm rawvaladdrhtk.list rawvaladdrwav.list

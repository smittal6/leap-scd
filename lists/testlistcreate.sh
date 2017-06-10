#This is the list creator for testing files
direc="/home/siddharthm/scd/wav/test"
#First create the raw.wav files
ls $direc $search_path > rawtestwav.list
#Then create the just raw names, would have to use sed
cp rawtestwav.list rawtestfiles.list
sed -i "s|\..*||g" rawtestfiles.list

cp rawtestwav.list rawtestaddrhtk.list
cp rawtestwav.list rawtestaddrwav.list
#addrhtk will contain the addr of htk files, addr wav will contain the addr of wav files[along with the files itself]
sed -i "s|^|/home/siddharthm/scd/feats/mfcc/test/|g" rawtestaddrhtk.list
sed -i "s|^|/home/siddharthm/scd/wav/test/|g" rawtestaddrwav.list

#Now let us make the final list required for HTK generation, which requires first wav file and then htk file
sed -i "s|wav|htk|g" rawtestaddrhtk.list #replacing the wav extension by htk extension
paste rawtestaddrwav.list rawtestaddrhtk.list > htkfinaltest.list
rm rawtestaddrhtk.list rawtestaddrwav.list

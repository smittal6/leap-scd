
#!/bin/bash
path_1='/home/siddharthm/TIMIT/'
fmt_1='.list'
fmt_2='*.WAV'
#fname1='train_files'
#fname1='val_files'
fname1='test_files'

while IFS='' read -r line || [[ -n "$line" ]]; do
	echo "Text read from file: $line"
	#echo $path_1
	#echo $line$fmt_2
	#find $path_1 -name $line -type d
	a1=$(find $path_1 -name $line -type d)
    a2=$(basename "$a1") #this is the speaker essentially
    arr=$(find $a1 -name $fmt_2 -exec basename {} ';') #This has all the sentences. Filter out SA1 and SA2
    for word in $arr
    do
            if [ "$word" = "SA1.WAV" ] || [ "$word" = "SA2.WAV" ]
            then
                    continue
            else
                    a3=$(echo $word | sed "s#.WAV##")
                    echo $a2"_"$a3 >> $fname1$fmt_1
            fi
    done
    #a4=$(find $a1 -name $fmt_2 -exec basename {} ';' | sed "s#^#$a2_#g") #This has speaker_sentence
    #find $a1 -name $fmt_2 >> $fname1$fmt_1
done < "$1"


# Important: Right now the configuration for file is set for generating the gamma features combined

#for i in $(seq 0 9); do
        #python combine_train.py x0$i $i &
        #echo $i
#done

#for i in $(seq 0 4); do
        #python combine_test.py x0$i $i &
        #python combine_val.py x0$i $i &
#done

#The combine_gamma program takes as input 3 arguments

#### SECTION for combining fbank features
#python combine_fbank.py train/ rawtrainfiles.list train &
#python combine_fbank.py test/ rawtestfiles.list test &
#python combine_fbank.py val/ rawvalfiles.list val &

#### SECTION for combining gamma features
python combine_gamma.py train/ rawtrainfiles.list train &
#python combine_gamma.py test/ rawtestfiles.list test &
python combine_gamma.py val/ rawvalfiles.list val &

#### SECTION for combining mfcc features
#python combine_only_mfcc.py train/ rawtrainfiles.list train &
#python combine_only_mfcc.py val/ rawvalfiles.list val &
#python combine_only_mfcc.py test/ rawtestfiles.list test &

wait
echo "Done nigga"

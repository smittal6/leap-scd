HTKDIR="/home/neerajs/Desktop/htk/HTKTools"

## Takes as input the list of files, where column 1 is the list of wav files, and the second column is the list of files where the features are going to be stored

$HTKDIR/HCopy -A -D -T 5 -C mfcc_config.cfg -S $1


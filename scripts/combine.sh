for i in $(seq 0 9); do
        python combine_train.py x0$i $i &
        #echo $i
done

for i in $(seq 0 4); do
        python combine_test.py x0$i $i &
        python combine_val.py x0$i $i &
done


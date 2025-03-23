#!/bin/bash


for i in {1..100}
do

filename="${i}_tran_curve.log"
if [ -f $filename ]; then 
cat $filename >> tran.log
echo "$filename is OK"
fi

done

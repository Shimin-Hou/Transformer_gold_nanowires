#!/bin/bash

for b in {1..67}
do
cd $b
#num_files=`ls -l|grep "^-"|wc -l|awk '{print int($0)}'`
num_files=`ls -l|grep "dat"|wc -l`
echo "OK"
echo $num_files

for a in $(seq 1 ${num_files})
do
  echo $a
  filename="~/Transformer/soap/soap_dat/${b}/${a}_soap_mod.dat"
  pythonfile_name="${a}_python.py" 
  npypath="~/Transformer/soap/soap_npy/${b}/${a}"
  cat "../soap_generator_test.py" $filename "../soap_generator_test2.py" > $pythonfile_name
  python $pythonfile_name $npypath
done
cd ..
done
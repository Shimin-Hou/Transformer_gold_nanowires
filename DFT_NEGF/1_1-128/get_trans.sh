#!/bin/bash
for a in {1..128}
do
  cd $a
  cat 0.tran.TRC | awk '{print $2}' | sed -n '306p' >> ~/Transformer/trans/1_tran_curve.log
  cd ..
done

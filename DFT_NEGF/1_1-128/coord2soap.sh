#!/bin/sh

for a in {1..128}
do
   fdffile=$a'_coord.fdf'
   outfile=$a'_soap_mod.dat'
   awk 'BEGIN{Nline=0}\
        {Nline++;x[Nline]=$1;y[Nline]=$2;z[Nline]=$3}\
        END{for(i=1;i<=Nline;i++){if(i==1){print "positions=[( ",x[i],",",y[i],",",z[i],")," } else if(i==Nline){print "(",x[i],",",y[i],",",z[i],")]"}\
        else print "( ",x[i],",",y[i],",",z[i],")," }} ' $a/$fdffile > ~/Transformer/soap/soap_dat/1/$outfile
done

import numpy as np
from os.path import exists
merge = []
for st in range(1,67+1):
    
        
    num=300
    for i in range(1,num+1):
        npypath = str(st)+'/'+str(i)+'_soap_result.npy'
        if(exists(npypath)):
            npyfile = np.load(npypath)
       # npyfile = np.reshape(npyfile,(1,304,1530))
            merge.append(npyfile)
        
merge = np.array(merge)
print(merge.shape)
np.save('struct_655.npy',merge) #the soap parameter rcut
        

import numpy as np

with open('tran.log','r') as fp:
    lines = fp.readlines()
merge=[]
for line in lines:
    merge.append(float(line))
merge=np.array(merge)
print(merge.shape)
np.save('tran.npy',merge)


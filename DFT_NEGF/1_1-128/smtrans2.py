import numpy as np
import os
import linecache
import subprocess

#import subprocess
def runcmd(command):
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=10)
    if ret.returncode == 0:
        print("success:",ret)
    else:
        print("error:",ret)

filename="1_1-128.atom"
file = open(filename)

file_num=0
coord_start=0

latx=linecache.getline(filename, 6)
line_x = latx.split();
lat_a = float(line_x[1])-float(line_x[0])
laty=linecache.getline(filename, 7)
line_y = laty.split();
lat_b = float(line_y[1])-float(line_y[0])
latz=linecache.getline(filename, 8)
line_z = latz.split();
lat_c = float(line_z[1])-float(line_z[0])
lattic = np.zeros((3,3))
lattic[0][0]=lat_a
lattic[1][1]=lat_b
lattic[2][2]=lat_c
#print(lattic)

total_num=linecache.getline(filename, 4)
total_num=int(total_num)

atom_num=1


leadright = np.zeros((1,10000)) 

while 1:
    line = file.readline()
    if not line:
        break
    if (line=="ITEM: TIMESTEP\n"): 
        file_num+=1
        if not os.path.exists(str(file_num)) : os.mkdir(str(file_num))
        f = open(str(file_num)+'/'+str(file_num)+"_coord_tmp"+".fdf",'w')
        f2= open(str(file_num)+'/'+str(file_num)+"_box"+".fdf",'w')
        #print("%block AtomicCoordinatesAndAtomicSpecies",file=f)
        zmin = 999999
        zmax = 0
        all_coord=np.zeros((total_num,3))
    if (coord_start==1):
        line_s = line.split()
        atom_coord=np.zeros((1,3))
        atom_coord[0][0]=float(line_s[2])
        atom_coord[0][1]=float(line_s[3])
        atom_coord[0][2]=float(line_s[4])
        atom_coord_car=np.dot(atom_coord,lattic)
        if (atom_coord_car[0][2]>zmax):zmax=atom_coord_car[0][2]
        if (atom_coord_car[0][2]<zmin):zmin=atom_coord_car[0][2]
        all_coord[atom_num-1]=atom_coord_car[0]
#        print("   ","%.8f" % atom_coord_car[0][0],"%.8f" % atom_coord_car[0][0],"%.8f" % atom_coord_car[0][0],"1",atom_num,"Au",file=f)
        atom_num+=1
        if (atom_num > total_num):
            lattic_z=zmax-zmin+2.08998832
            leadright[0][file_num]=lattic_z-8.35995328
            for i in range(total_num):
               atom_coord_z=all_coord[i][2]-zmin
               print("   ","%.8f" % all_coord[i][0],"%.8f" % all_coord[i][1],"%.8f" % atom_coord_z, file=f)
            #print("%endblock AtomicCoordinatesAndAtomicSpecies",file=f)
            print("%block LatticeVectors",file=f2)
            print(" ","%.8f" % lat_a,"%.8f" % 0,"%.8f" % 0,file=f2)
            print(" ","%.8f" % 0,"%.8f" % lat_b,"%.8f" % 0,file=f2)
            print(" ","%.8f" % 0,"%.8f" % 0,"%.8f" % lattic_z,file=f2)
            print("%endblock LatticeVectors",file=f2)
            coord_start=0
            atom_num=1
            f.close()
            f2.close()
   
            
    
        
    if (line=="ITEM: ATOMS id type xs ys zs\n"):
        coord_start=1     
print(leadright)
for i in range(file_num):
  f3= open(str(i+1)+'/'+str(i+1)+"_rlead_z"+".fdf",'w')
  for k in range(4):
    for j in range(25):
        rlead_z=leadright[0][i+1]+2.08998832*k
        print(" ","%.8f" % rlead_z,file=f3)
  f3.close()

for i in range(1,file_num+1):
         runcmd("sort -t $' ' -k7 -n ./"+str(i)+"/"+str(i)+"_coord_tmp.fdf >./"+str(i)+"/"+str(i)+"_coord_tmp2.fdf")
         runcmd("sed '205,$d' -i ./"+str(i)+"/"+str(i)+"_coord_tmp2.fdf")
         runcmd("sed '1,100d' -i ./"+str(i)+"/"+str(i)+"_coord_tmp2.fdf")
         runcmd("paste ./"+str(i)+"/"+str(i)+"_coord_tmp2.fdf ./Au_num_tmp.fdf > ./"+str(i)+"/"+str(i)+"_coord_tmp3.fdf")
         runcmd("paste ./lead_temp_r.fdf ./"+str(i)+"/"+str(i)+"_rlead_z.fdf ./Au_r_num_temp.fdf > ./"+str(i)+"/"+str(i)+"_rightlead.fdf")
         #runcmd("paste ./lead_temp_r.fdf ./Au_r_num_temp.fdf> "+str(i)+"/"+str(i)+"_rightlead.fdf")
         runcmd("cat ./lead_tmp.fdf > ./"+str(i)+"/"+str(i)+"_coord.fdf")
         runcmd("cat ./"+str(i)+"/"+str(i)+"_coord_tmp3.fdf >> ./"+str(i)+"/"+str(i)+"_coord.fdf")
         runcmd("cat ./"+str(i)+"/"+str(i)+"_rightlead.fdf >> ./"+str(i)+"/"+str(i)+"_coord.fdf")
         runcmd("sed -i '$d' ./"+str(i)+"/"+str(i)+"_coord.fdf" )
         runcmd("cat ./"+str(i)+"/"+str(i)+"_box.fdf > ./"+str(i)+"/"+str(i)+".fdf")
         runcmd("echo '%block AtomicCoordinatesAndAtomicSpecies' >> ./"+str(i)+"/"+str(i)+".fdf")
         runcmd("cat ./"+str(i)+"/"+str(i)+"_coord.fdf >> ./"+str(i)+"/"+str(i)+".fdf")
         runcmd("echo '%endblock AtomicCoordinatesAndAtomicSpecies' >> ./"+str(i)+"/"+str(i)+".fdf")
         runcmd("cp ./temp/* ./"+str(i))
         runcmd("cat ./"+str(i)+"/"+str(i)+".fdf >> ./"+str(i)+"/tran.fdf")    

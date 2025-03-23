import numpy as np
from model import DNNnet,TSnet_small
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.utils.data as data
#from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter('./logs')


dataset = np.load('soap/soap_npy/struct_655.npy').reshape(7319,1,304,1530)
target = np.load('trans/tran_new.npy')/10

print("Done:loading data")

dataset = torch.from_numpy(dataset).type(torch.cuda.FloatTensor)
target = torch.from_numpy(target).type(torch.cuda.FloatTensor)

torch_dataset= data.TensorDataset(dataset, target)
train_db,val_db,test_db = torch.utils.data.random_split(torch_dataset, [6719,100,500])

train_loader = data.DataLoader(dataset=train_db, batch_size=32,shuffle=True)
val_loader = data.DataLoader(dataset=val_db, batch_size=32,shuffle=True)
test_loader = data.DataLoader(dataset=test_db, batch_size=1,shuffle=True)


model = TSnet_small().cuda()

LR = 1E-5
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
#loss_func = nn.MSELoss()
loss_func = nn.HuberLoss(delta=1.0)
EPOCH=100000

loss_value=[]
for epoch in range(EPOCH):
    f2=open('tf_log.log','a')
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x).cuda()
        batch_y = Variable(y).cuda()
        output = model(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value.append(loss.item())
    for j, (xv, yv) in enumerate(val_loader):
        batch_xv = Variable(xv)
        batch_yv = Variable(yv)
        output = model(batch_xv)
        val_loss = loss_func(output, batch_yv)
    print('Epoch: ',epoch,'  Loss: ',loss.item(), '                  Val_loss: ',val_loss.item(),file=f2)
    if (epoch%100 == 0):
        model.eval()
        testfilename = './test_tf/result_tf_epoch_'+str(epoch)+".log"
        f = open(testfilename,'w')
        pre = np.zeros([500,])
        true_val = np.zeros([500,])
        for i, (x, y) in enumerate(test_loader):
            batch_x = Variable(x)
            batch_y = Variable(y)
            output = model(batch_x)
            pre[i]=output
            true_val[i]=batch_y
            print(float(output),"         ",float(batch_y),file=f)
        f.close()
        print(epoch,np.corrcoef(pre,true_val),file=f2)
        modelname = "./checkpoints/cp_tf_epoch_"+str(epoch)+".pt"
        torch.save(model.state_dict(), modelname)
        model.train()
    f2.close()



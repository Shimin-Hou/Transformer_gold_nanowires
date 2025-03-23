import torch
import torch.nn as nn




class DNNnet(torch.nn.Module):
    def __init__(self):
        super(DNNnet,self).__init__()
        self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(1530,64),
                    torch.nn.ELU())
        self.mlp2 = torch.nn.Sequential(
                    torch.nn.Linear(64,128),
                    torch.nn.ELU())
        self.mlp3 = torch.nn.Sequential(
                    torch.nn.Linear(128,256),
                    torch.nn.ELU())
        self.mlp4 = torch.nn.Sequential(
                    torch.nn.Linear(256,256),
                    torch.nn.ELU())
        self.mlp5 = torch.nn.Sequential(
                    torch.nn.Linear(256,256),
                    torch.nn.ELU())
        self.mlp6 = torch.nn.Sequential(
                    torch.nn.Linear(256,64),
                    torch.nn.ELU())
        self.mlp7 = torch.nn.Sequential(
                    torch.nn.Linear(64,64),
                    torch.nn.ELU())           
        self.mlp8 = torch.nn.Sequential(
                    torch.nn.Linear(64,64),
                    torch.nn.ELU())
        self.mlp9 = torch.nn.Sequential(
                    torch.nn.Linear(64,1))                         
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,256), return_indices= False)
    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.gmp(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x

class DNNnet_dropout(torch.nn.Module):
    def __init__(self):
        super(DNNnet_dropout,self).__init__()
        self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(1530,64),
                    torch.nn.ELU())
        self.mlp2 = torch.nn.Sequential(
                    torch.nn.Linear(64,128),
                    torch.nn.Dropout(0.1),
                    torch.nn.ELU())
        self.mlp3 = torch.nn.Sequential(
                    torch.nn.Linear(128,256),
                    torch.nn.Dropout(0.1),
                    torch.nn.ELU())
        self.mlp4 = torch.nn.Sequential(
                    torch.nn.Linear(256,256),
                    torch.nn.Dropout(0.1),
                    torch.nn.ELU())
        self.mlp5 = torch.nn.Sequential(
                    torch.nn.Linear(256,256),
                    torch.nn.ELU())
        self.mlp6 = torch.nn.Sequential(
                    torch.nn.Linear(256,64),
                    torch.nn.ELU())
        self.mlp7 = torch.nn.Sequential(
                    torch.nn.Linear(64,64),
                    torch.nn.ELU())           
        self.mlp8 = torch.nn.Sequential(
                    torch.nn.Linear(64,64),
                    torch.nn.ELU())
        self.mlp9 = torch.nn.Sequential(
                    torch.nn.Linear(64,1))                         
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,256), return_indices= False)
    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.gmp(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x

class DNNnet_huge(torch.nn.Module):
    def __init__(self):
        super(DNNnet_huge,self).__init__()
        self.mlp1 = torch.nn.Sequential(
                    torch.nn.Linear(1530,256),
                    torch.nn.ELU())
        self.mlp2 = torch.nn.Sequential(
                    torch.nn.Linear(256,2048),
                    torch.nn.ELU())
        self.mlp3 = torch.nn.Sequential(
                    torch.nn.Linear(2048,256),
                    torch.nn.ELU())
        self.mlp4 = torch.nn.Sequential(
                    torch.nn.Linear(256,256),
                    torch.nn.ELU())
        self.mlp5 = torch.nn.Sequential(
                    torch.nn.Linear(256,256),
                    torch.nn.ELU())
        self.mlp6 = torch.nn.Sequential(
                    torch.nn.Linear(256,64),
                    torch.nn.ELU())
        self.mlp7 = torch.nn.Sequential(
                    torch.nn.Linear(64,64),
                    torch.nn.ELU())           
        self.mlp8 = torch.nn.Sequential(
                    torch.nn.Linear(64,64),
                    torch.nn.ELU())
        self.mlp9 = torch.nn.Sequential(
                    torch.nn.Linear(64,1))                         
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,256), return_indices= False)
    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.gmp(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x

class TSnet(torch.nn.Module):
    def __init__(self):
        super(TSnet,self).__init__()
        #self.bn = torch.nn.BatchNorm2d(1)
        self.tf =nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.te = nn.TransformerEncoder(self.tf, num_layers=6)
        self.mlp = torch.nn.Linear(1024,256)
        self.mlp2 = torch.nn.Linear(1530,1024)
        self.mlp6 = torch.nn.Sequential(
            torch.nn.Linear(256,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp7 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp_add3 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())

        self.mlp_add4 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())            

            
        self.mlp8 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
        self.mlp9 = torch.nn.Sequential(
            torch.nn.Linear(64,1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,1024), return_indices= False)
    def forward(self, x):
        #x = self.bn(x)
        x = self.mlp2(x)
        x = self.te(x)
        x = self.gmp(x)
        x = self.mlp(x)
        x = self.mlp6(x)
        #x = self.mlp4(x.view(x.size(0),-1))
        x = self.mlp7(x)
        # x = self.mlp_add3(x)
        # x = self.mlp_add4(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x

class TSnet_small(torch.nn.Module):
    def __init__(self):
        super(TSnet_small,self).__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.tf =nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.te = nn.TransformerEncoder(self.tf, num_layers=2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256,256),
            torch.nn.ReLU())   
        self.mlp2 = torch.nn.Linear(1530,256)
        self.mlp6 = torch.nn.Sequential(
            torch.nn.Linear(256,64),
            torch.nn.ReLU())           
        self.mlp7 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ReLU())            
        self.mlp_add3 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ReLU())
        self.mlp_add4 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ReLU())                       
        self.mlp8 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ReLU())
        self.mlp9 = torch.nn.Sequential(
            torch.nn.Linear(64,1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,256), return_indices= False)
    def forward(self, x):
        x = self.mlp2(x)
        x = self.te(x)
        x = self.mlp(x)
        x = self.gmp(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x
        
class TSnet_small_org(torch.nn.Module):
    def __init__(self):
        super(TSnet_small_org,self).__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.tf =nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.te = nn.TransformerEncoder(self.tf, num_layers=2)
        self.mlp = torch.nn.Linear(256,256)
        self.mlp2 = torch.nn.Linear(1530,256)
        self.mlp6 = torch.nn.Sequential(
            torch.nn.Linear(256,64),
            torch.nn.ReLU())           
        self.mlp7 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ReLU())            
        self.mlp_add3 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ReLU())
        self.mlp_add4 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ReLU())                       
        self.mlp8 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ReLU())
        self.mlp9 = torch.nn.Sequential(
            torch.nn.Linear(64,1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,256), return_indices= False)
    def forward(self, x):
        x = self.bn(x)
        x = x.squeeze(1)
        x = self.mlp2(x)
        x = self.te(x)
        x = self.mlp(x)
        x = self.gmp(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x

class TSnet_small_1layer(torch.nn.Module):
    def __init__(self):
        super(TSnet_small_1layer,self).__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.tf =nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.te = nn.TransformerEncoder(self.tf, num_layers=1)
        self.mlp = torch.nn.Linear(256,256)
        self.mlp2 = torch.nn.Linear(1530,256)
        self.mlp6 = torch.nn.Sequential(
            torch.nn.Linear(256,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp7 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp_add3 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())

        self.mlp_add4 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ELU())            

            
        self.mlp8 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
            torch.nn.ELU())
        self.mlp9 = torch.nn.Sequential(
            torch.nn.Linear(64,1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,256), return_indices= False)
    def forward(self, x):
        x = self.bn(x)
        x = x.squeeze(1)
        x = self.mlp2(x)
        x = self.te(x)
        x = self.mlp(x)
        x = self.gmp(x)
        x = self.mlp6(x)
        #x = self.mlp4(x.view(x.size(0),-1))
        x = self.mlp7(x)
        # x = self.mlp_add3(x)
        # x = self.mlp_add4(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x

class TSnet_small_4layer(torch.nn.Module):
    def __init__(self):
        super(TSnet_small_4layer,self).__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.tf =nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.te = nn.TransformerEncoder(self.tf, num_layers=4)
        self.mlp = torch.nn.Linear(256,256)
        self.mlp2 = torch.nn.Linear(1530,256)
        self.mlp6 = torch.nn.Sequential(
            torch.nn.Linear(256,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp7 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp_add3 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())

        self.mlp_add4 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())            

            
        self.mlp8 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
        self.mlp9 = torch.nn.Sequential(
            torch.nn.Linear(64,1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,256), return_indices= False)
    def forward(self, x):
        x = self.bn(x)
        x = x.squeeze(1)
        x = self.mlp2(x)
        x = self.te(x)
        x = self.mlp(x)
        x = self.gmp(x)
        x = self.mlp6(x)
        #x = self.mlp4(x.view(x.size(0),-1))
        x = self.mlp7(x)
        # x = self.mlp_add3(x)
        # x = self.mlp_add4(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x


class TSnet_small_cls(torch.nn.Module):
    def __init__(self):
        super(TSnet_small,self).__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.tf =nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.te = nn.TransformerEncoder(self.tf, num_layers=2)
        self.mlp = torch.nn.Linear(256,256)
        self.mlp2 = torch.nn.Linear(1530,256)
        self.mlp6 = torch.nn.Sequential(
            torch.nn.Linear(256,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp7 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp_add3 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())

        self.mlp_add4 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())            

            
        self.mlp8 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
        self.mlp9 = torch.nn.Sequential(
            torch.nn.Linear(64,1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,256), return_indices= False)
    def forward(self, x):
        x = self.bn(x)
        x = x.squeeze(1)
        x = self.mlp2(x)
        x = self.te(x)
        x = self.mlp(x)
        x = x[:,0,:].reshape(-1,256)
        x = self.mlp6(x)
        #x = self.mlp4(x.view(x.size(0),-1))
        x = self.mlp7(x)
        # x = self.mlp_add3(x)
        # x = self.mlp_add4(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x

class TSnet_small_ELU(torch.nn.Module):
    def __init__(self):
        super(TSnet_small_ELU,self).__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.tf =nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.te = nn.TransformerEncoder(self.tf, num_layers=2)
        self.mlp = torch.nn.Linear(256,256)
        self.mlp2 = torch.nn.Linear(1530,256)
        self.mlp6 = torch.nn.Sequential(
            torch.nn.Linear(256,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp7 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
            
        self.mlp_add3 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())

        self.mlp_add4 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())            

            
        self.mlp8 = torch.nn.Sequential(
            torch.nn.Linear(64,64),
#            torch.nn.ReLU()
            torch.nn.ELU())
        self.mlp9 = torch.nn.Sequential(
            torch.nn.Linear(64,1))
        self.gmp = torch.nn.AdaptiveMaxPool2d((1,256), return_indices= False)
    def forward(self, x):
        x = self.bn(x)
        x = x.squeeze(1)
        x = self.mlp2(x)
        x = self.te(x)
        x = self.mlp(x)
        x = self.gmp(x)
        x = self.mlp6(x)
        #x = self.mlp4(x.view(x.size(0),-1))
        x = self.mlp7(x)
        # x = self.mlp_add3(x)
        # x = self.mlp_add4(x)
        x = self.mlp8(x)
        x = self.mlp9(x)
        x = x.reshape(-1,)
        return x
'''
    These code are written By Mehdi Touyserkani.
    Email Address: ir_bestrpo@yahoo.com.
    Personal Website: Https://www.ir-bestpro.com
    LinkedIn: https://www.linkedin.com/in/bestpro-group/
    
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np

from Parameters import Params
from Creatre_Statistical_Features import Statistical_Features

#__________________Encoder Class__________________________

class Encoder(nn.Module):
    def __init__(self , in_channels):
        super(Encoder, self).__init__()
        self.Coeffs = []
        self.layer1 = nn.Sequential(nn.Linear(in_channels,in_channels) , nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(in_channels,in_channels) , nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(in_channels,in_channels) , nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Linear(in_channels,in_channels) , nn.LeakyReLU())
        self.layer5 = nn.Linear(in_channels,in_channels)
    
    def forward(self , random_noise):
        out = self.layer1(random_noise)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return F.leaky_relu(out)
    
#______Start DWT Feature Extractor and Deep Learning_________

    def Extract_Encoder_Features(self):

        Dataset = pd.read_csv('./Features/Denoised.CSV').iloc[: , 1:].to_numpy() # Convert To Numpy Array

        #__________Create VGGNET Instances__________

        optimizer = torch.optim.SGD(self.parameters() , Params.LEARNING_RATE , Params.MOMENTUM , Params.WEIGHT_DECAY) # Using SGD Weight Optimizer
        Loss = nn.MSELoss() # Using MSE Loss Function

        #_________Extract Coefficients_______________

        for epoch in range(1): # Running Model in N Epoch
            temp_loss = 0 # Temp Loss in each Epoch
            for batch in range(0 , len(Dataset) , 1): # Reading all Coefficients in serveral Batches (80% For Train)

                X = torch.from_numpy(Dataset[batch:batch + Params.BATCH_SIZE]).float() 
                coef = torch.sqrt(torch.sum(X**2)) # calc signal coefficients
                X = X / (coef) # Data Normilization
                random_noise = torch.randn(len(X) , 4097) # The latent variable in Encoder model
                output = self.forward(random_noise) # Calling the Model (Encoder)

                loss = Loss(output , X) # Calc Loss Function
                temp_loss = loss.item() 

                self.zero_grad() # Don't Save Gradient History
                loss.backward() # BackPropagation Process
                optimizer.step() # Update Weights

                self.Coeffs.append (Statistical_Features([np.mean(self.layer5.weight.detach().numpy(),axis=1)]).Create_Features()) # append encoder coeffs to coeffiecients array of each signal

                print('Signal # ' , str(batch + 1) , ', Reconstruction Loss : ' , str(temp_loss))

model = Encoder(4097) # Create Encoder Model object instance
model.Extract_Encoder_Features() # start encoder features extracttion
pd.DataFrame(model.Coeffs).to_csv('./Features/Encoder_Features.csv')

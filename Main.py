'''
    These code are written By Mehdi Touyserkani.
    Email Address: ir_bestrpo@yahoo.com.
    Personal Website: Https://www.ir-bestpro.com
    LinkedIn: https://www.linkedin.com/in/bestpro-group/
    
'''

import numpy as np
import pandas as pd
import torch.optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

#__________Import Other Directories______________________

from Parameters import Params

class Dense_Network(nn.Module):

    def __init__(self , features_length , number_of_clesses):
        super(Dense_Network , self).__init__()
        self.FC1 = nn.Sequential(nn.Linear(features_length,256) , nn.BatchNorm1d(256) , nn.ReLU())
        self.FC2 = nn.Sequential(nn.Linear(256,512) , nn.BatchNorm1d(512) ,  nn.ReLU())
        self.FC3 = nn.Sequential(nn.Linear(512,1024) , nn.BatchNorm1d(1024) ,  nn.ReLU())
        self.FC4 = nn.Sequential(nn.Linear(1024,number_of_clesses))

    def forward(self , x):
        out = self.FC1(x)
        out = self.FC2(out)
        out = self.FC3(out)
        out = self.FC4(out)
        return out

#______Start DWT Feature Extractor and Deep Learning_________

if __name__ == '__main__':
    
    DWT_EMD_Features = pd.read_csv('./Features/DWT_EMD_Features.CSV').iloc[:,1:].to_numpy() # Reading DWT_EMD Statistical Features
    Decoder_Features = pd.read_csv('./Features/Encoder_Features.CSV').iloc[:,1:].to_numpy() # Reading Auto-Encoder Statistical Features
    Dataset = np.column_stack((DWT_EMD_Features , Decoder_Features)) # Concatenation of Encoder coefficients and DWT_EMD coefficients
    Labels = pd.read_csv('./Features/Labels.CSV').iloc[:,1].to_numpy(dtype='float') # Reading Labels

    X_train, X_test, y_train, y_test = train_test_split(Dataset, Labels, test_size=0.2, random_state=42) # Splitting Train and Test Frequencies

    X_test = torch.from_numpy(X_test).float() # convert to Tensor
    y_test = torch.from_numpy(y_test).float() # convert to Tensor

    #______________Create Dense_Network Instances_____________________

    model = Dense_Network( Dataset.shape[1] , 5) # Create Dense_Network Instance
    optimizer = torch.optim.Adam(model.parameters() , Params.LEARNING_RATE) # Using Adam Weight Optimizer
    Loss = nn.CrossEntropyLoss() # Using Cross Entropy Loss Function

    #_______________Train Phase__________________

    for epoch in range(Params.EPOCHS): # Running Model in N Epoch
        temp_loss = 0 # Temp Loss in each Epoch
        steps = 0 # Counting Number of Batches
        temp_acc = 0
        for batch in range(0 , len(X_train) , Params.BATCH_SIZE): # Reading all Coefficients in serveral Batches (80% For Train)

            X = torch.from_numpy(X_train[batch : (batch + Params.BATCH_SIZE) , :]).float() # Reading Current Coefficients Vectors
            targets = torch.from_numpy(y_train[batch : (batch + Params.BATCH_SIZE)]).float() # Reading DataLabels
            output = model(X) # Calling the Model and get generated outputs
            loss = Loss(output, targets.long()) # Calc Loss Function
            temp_loss += loss.item()
            steps+=1

            temp_acc += (((torch.argmax(output.data , 1).float() == targets).sum().item()) / len(targets)) 

            model.zero_grad() # Don't Save Gradient History
            loss.backward() # BackPropagation Process
            optimizer.step() # Update Weights

        print('Train Phase - Epoch # ' , str(epoch + 1) , ', Loss : ' , str(temp_loss / steps))
    
    #________________Test Phase____________________

        with torch.no_grad(): # Stop Weight Updating
            output = model(X_test) # Calling the Model
            output = torch.argmax(output.data , 1).float() # works as a softmax activation function
            corrected = (output == y_test).sum().item() # number of correct items      
            print('Test Phase - The Model Accuracy :' , str(np.mean((corrected/len(X_test) , (temp_acc / steps))) *100) + '%')



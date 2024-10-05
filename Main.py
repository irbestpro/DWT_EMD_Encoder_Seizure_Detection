'''
    These code are written By Mehdi Touyserkani.
    Email Address: ir_bestrpo@yahoo.com.
    Personal Website: Https://www.ir-bestpro.com
    LinkedIn: https://www.linkedin.com/in/bestpro-group/
    
'''

import math
import numpy as np
import pandas as pd
import torch.optim
import torch.nn as nn
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from transformers import get_scheduler

#_____________Import Other Files______________________

from Parameters import Params

class Dense_Network(nn.Module):

    def __init__(self , features_length , number_of_clesses):
        super(Dense_Network , self).__init__()
        self.FC1 = nn.Sequential(nn.Linear(features_length , 256) , nn.LeakyReLU())
        self.FC2 = nn.Sequential(nn.Linear(256 , 512) , nn.LeakyReLU())
        self.FC3 = nn.Sequential(nn.Linear(512 , 2048) , nn.LeakyReLU())
        self.FC4 = nn.Sequential(nn.Linear(2048 , number_of_clesses))

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

    X_train, X_test, y_train, y_test = train_test_split(Dataset, Labels, test_size=0.25, random_state=42) # Splitting Train and Test Frequencies

    X_test = torch.from_numpy(X_test).float() # convert to Tensor
    y_test = torch.from_numpy(y_test).float() # convert to Tensor

    #______________Create Dense_Network Instances_____________________

    model = Dense_Network(Dataset.shape[1] , 5) # Create Dense_Network Instance
    best_model = None
    optimizer = torch.optim.Adam(model.parameters() , lr = Params.LEARNING_RATE , weight_decay = Params.WEIGHT_DECAY) # Using Adam Weight Optimizer
    scheduler = get_scheduler(name="linear" , optimizer=optimizer , num_warmup_steps=0 , num_training_steps = len(X_train) * Params.EPOCHS)
    Loss = nn.CrossEntropyLoss() # Using Cross Entropy Loss Function

    #_______________Train Phase__________________

    Trainings = []
    Tests = []
    best_performance = - math.inf
    for epoch in range(1,Params.EPOCHS): # Running Model in N Epoch
        temp_loss = 0 # Temp Loss in each Epoch
        steps = 0 # Counting Number of Batches
        temp_acc = 0
        model.train() # Training Phase of Model
        for batch in range(0 , len(X_train) , Params.BATCH_SIZE): # Reading all Coefficients in serveral Batches (80% For Train)

            X = torch.from_numpy(X_train[batch : (batch + Params.BATCH_SIZE) , :]).float() # Reading Current Coefficients Vectors
            targets = torch.from_numpy(y_train[batch : (batch + Params.BATCH_SIZE)]).type(torch.LongTensor) # Reading DataLabels
            output = model(X) # Calling the Model and get generated outputs
            loss = Loss(output, targets) # Calc Loss Function
            temp_loss += loss.item()
            steps+=1

            #temp_acc += (((torch.argmax(output.data , 1).float() == targets).sum().item()) / len(targets)) 
            temp_acc += accuracy_score(targets , (torch.argmax(output.data , 1).float()))

            model.zero_grad() # Don't Save Gradient History
            loss.backward() # BackPropagation Process
            optimizer.step() # Update Weights
            scheduler.step()
    
    #________________Test Phase____________________

        model.eval() # Evaluation Phase
        with torch.no_grad(): # Stop Weight Updating
            output = model(X_test) # Calling the Model
            output = torch.argmax(output.data , 1).float() # works as a softmax activation function
            test_performance = accuracy_score(y_test , output) # (output == y_test).sum().item() # number of correct items      
            Trainings.append(temp_acc / steps)
            Tests.append(test_performance)
            print(f'{epoch =} - Test : {test_performance * 100}% , Training : {Trainings[-1]}% , Loss: {(temp_loss / steps)}')

            #___________Model Checkpoints_____________

            if best_performance < test_performance:
                best_model = deepcopy(model)
                best_performance = test_performance

    #_____________Final Evaluation____________________

    model.eval()
    with torch.no_grad():
        output = best_model(X_test) # Calling the Model
        output = torch.argmax(output.data , 1).float() # works as a softmax activation function
        corrected = accuracy_score(y_test , output)
        print(f'The Model Accuracy is: {corrected * 100}')
        print(confusion_matrix(y_test , output))

    #______________Plot Results_______________________

    plt.plot(range(0,len(Trainings)), Trainings , label = "Training Phase")
    plt.plot(range(0,len(Tests)), Tests , label = "Test Phase")
    plt.title('EEG Signal Classification Model')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.show()



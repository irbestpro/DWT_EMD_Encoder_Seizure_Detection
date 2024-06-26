'''
    These code are written By Mehdi Touyserkani.
    Email Address: ir_bestrpo@yahoo.com.
    Personal Website: Https://www.ir-bestpro.com
    LinkedIn: https://www.linkedin.com/in/bestpro-group/
    
'''

import numpy as np
from scipy.stats import skew,kurtosis

class Statistical_Features():

    def __init__(self , Data): # Class Constructor
        self.Data = Data
        self.X = []

    def Create_Features(self):
        for i in range(0,len(self.Data)):
            temp = self.Data[i]
            self.X.append(np.mean(temp))
            self.X.append(skew(temp , bias=True))
            self.X.append(kurtosis(temp, bias=True))
            self.X.append(np.sqrt(np.mean(temp**2)))
            
            #self.X.append(np.std(temp))
            #self.X.append(np.sum(temp**2))
            #self.X.append(np.sqrt(np.mean(temp**2)) / np.sum(temp**2))
            #self.X.append(np.mean(temp) / np.std(temp))
        
        return self.X
